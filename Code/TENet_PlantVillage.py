import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ==========================================
# 1. TENet 核心算子：图化演化层（按 model_arch 重写）
# ==========================================
class TrainablePhi(nn.Module):
    """可学习的连接强度映射函数 phi"""

    def __init__(self, phi_temp_init=1.0, phi_residual_init=0.1):
        super().__init__()
        phi_temp_init = max(float(phi_temp_init), 1e-4)
        phi_residual_init = max(float(phi_residual_init), 1e-6)
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        self.temperature_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(phi_temp_init, dtype=torch.float32))))
        self.residual_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(phi_residual_init, dtype=torch.float32))))

    def forward(self, vi, vj):
        base = torch.abs(vi - vj)
        combined = torch.stack([vi, vj], dim=-1)  # (M, C, 2)
        logits = self.net(combined).squeeze(-1)   # (M, C)
        temperature = torch.nn.functional.softplus(self.temperature_raw) + 1e-4
        residual_weight = torch.nn.functional.softplus(self.residual_raw)
        phi_soft = torch.sigmoid(logits / temperature)
        return base + residual_weight * phi_soft


class AlgebraicGraphLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        chunk_size=2048,
        phi_temp_init=1.0,
        phi_residual_init=0.1,
        use_learnable_phi=True
    ):
        super(AlgebraicGraphLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels + 1
        self.chunk_size = chunk_size
        self.phi = TrainablePhi(phi_temp_init=phi_temp_init, phi_residual_init=phi_residual_init) if use_learnable_phi else None
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=10, bias=True)
        self.register_buffer('L', self._build_laplacian())

    def _build_laplacian(self):
        A = torch.zeros((10, 10))
        edges = [
            (0, 4), (1, 4),
            (0, 5), (2, 5),
            (0, 6), (3, 6),
            (1, 7), (2, 7),
            (1, 8), (3, 8),
            (2, 9), (3, 9)
        ]
        for i, j in edges:
            A[i, j] = A[j, i] = 1.0
        return torch.diag(A.sum(dim=1)) - A

    def _graphify_one_chunk(self, patch_chunk):
        # patch_chunk: [P, C, 4]
        v1, v2, v3, v4 = patch_chunk[:, :, 0], patch_chunk[:, :, 1], patch_chunk[:, :, 2], patch_chunk[:, :, 3]

        if self.phi is not None:
            e12, e13, e14 = self.phi(v1, v2), self.phi(v1, v3), self.phi(v1, v4)
            e23, e24, e34 = self.phi(v2, v3), self.phi(v2, v4), self.phi(v3, v4)
        else:
            e12, e13, e14 = torch.abs(v1 - v2), torch.abs(v1 - v3), torch.abs(v1 - v4)
            e23, e24, e34 = torch.abs(v2 - v3), torch.abs(v2 - v4), torch.abs(v3 - v4)

        z = torch.stack([v1, v2, v3, v4, e12, e13, e14, e23, e24, e34], dim=2)  # [P, C, 10]
        G = z.unsqueeze(3) * self.L * z.unsqueeze(2)  # [P, C, 10, 10]
        return G

    def forward(self, x):
        # x: [B, C_l, H_l, W_l]
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Channel mismatch: expected {self.in_channels}, got {c}")
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Spatial size must be even, got H={h}, W={w}")

        h2, w2 = h // 2, w // 2

        # 划分 2x2 非重叠组，每组每通道 4 个标量
        patches = x.view(b, c, h2, 2, w2, 2).permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(b * h2 * w2, c, 4)               # [P, C_l, 4]

        # 分块构图，避免显存峰值过高
        outputs = []
        for start in range(0, patches.size(0), self.chunk_size):
            end = min(start + self.chunk_size, patches.size(0))
            patch_chunk = patches[start:end]                        # [Pc, C_l, 4]

            # M_p ∈ R^(10x10xC_l)
            G = self._graphify_one_chunk(patch_chunk)              # [Pc, C_l, 10, 10]

            # 单层卷积直接提特征: R^(10x10xC_l) -> R^(C_l+1)
            y = self.conv(G).squeeze(-1).squeeze(-1)               # [Pc, C_l+1]
            outputs.append(y)

        y_all = torch.cat(outputs, dim=0)                          # [P, C_l+1]
        y_all = y_all.view(b, h2, w2, self.out_channels).permute(0, 3, 1, 2)
        # 输出 X^(l+1): [B, C_l+1, H_l/2, W_l/2]
        return y_all


# ==========================================
# 2. TENet 整体架构 (针对 PlantVillage 256x256 RGB)
# ==========================================
class TENet_PlantVillage(nn.Module):
    def __init__(
        self,
        num_classes=39,
        num_layers=4,
        chunk_size=2048,
        input_size=256,
        phi_temp_init=1.0,
        phi_residual_init=0.1,
        use_learnable_phi=True
    ):
        super(TENet_PlantVillage, self).__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1")
        if input_size % (2 ** num_layers) != 0:
            raise ValueError(
                f"input_size={input_size} is not divisible by 2^{num_layers}; "
                f"final spatial size must be integer."
            )

        self.num_layers = num_layers
        layers = []
        in_channels = 3
        for _ in range(num_layers):
            layers.append(
                AlgebraicGraphLayer(
                    in_channels=in_channels,
                    chunk_size=chunk_size,
                    phi_temp_init=phi_temp_init,
                    phi_residual_init=phi_residual_init,
                    use_learnable_phi=use_learnable_phi
                )
            )
            in_channels += 1
        self.layers = nn.ModuleList(layers)
        self.final_channels = in_channels
        self.final_spatial = input_size // (2 ** num_layers)

        # 分类头：不做池化，直接展平 C_final x H_final x W_final
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_channels * self.final_spatial * self.final_spatial, num_classes)
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        feat = x
        for layer in self.layers:
            feat = layer(feat)
        return self.classifier(feat)


# ==========================================
# 3. 数据与训练
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    data_root,
    image_size=256,
    batch_size=32,
    val_batch_size=64,
    test_batch_size=64,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    # 先创建一次拿到样本索引，再按索引构造子集
    full_dataset_for_split = datasets.ImageFolder(root=data_root)
    num_classes = len(full_dataset_for_split.classes)
    total_size = len(full_dataset_for_split)

    if num_classes != 39:
        print(f"Warning: detected {num_classes} classes (expected 39). Continue training with detected classes.")

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    if test_size <= 0:
        raise ValueError("Dataset split invalid. Please adjust train/val ratios.")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset_for_split,
        [train_size, val_size, test_size],
        generator=generator
    )

    # 分别用不同 transform 重建子集
    full_train = datasets.ImageFolder(root=data_root, transform=train_transform)
    full_eval = datasets.ImageFolder(root=data_root, transform=eval_transform)

    train_set = torch.utils.data.Subset(full_train, train_subset.indices)
    val_set = torch.utils.data.Subset(full_eval, val_subset.indices)
    test_set = torch.utils.data.Subset(full_eval, test_subset.indices)

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    print(
        f"Dataset ready: classes={num_classes}, total={total_size}, "
        f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}, workers={num_workers}"
    )
    return train_loader, val_loader, test_loader, full_dataset_for_split.classes


def evaluate(model, data_loader, device, use_amp=False):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    eval_iter = data_loader
    if tqdm is not None:
        eval_iter = tqdm(data_loader, desc="Valid/Test", leave=False)

    with torch.no_grad():
        for data, target in eval_iter:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                output = model(data)
                loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            if tqdm is not None:
                eval_iter.set_postfix(acc=f"{(100.0 * correct / max(total, 1)):.2f}%")

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train(data_root, epochs=20, batch_size=16, lr=1e-3, num_layers=4, chunk_size=2048, image_size=256, use_amp=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        val_batch_size=64,
        test_batch_size=64,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )

    model = TENet_PlantVillage(
        num_classes=len(class_names),
        num_layers=num_layers,
        chunk_size=chunk_size,
        input_size=image_size
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Num classes: {len(class_names)}")
    print(f"TENet layers: {num_layers}, channel rule: C_l -> C_l+1")
    print(f"Classifier input dim: {model.final_channels}x{model.final_spatial}x{model.final_spatial}")
    print(f"AMP enabled: {use_amp and device.type == 'cuda'}")

    best_val_acc = 0.0
    best_model_path = os.path.join(os.path.dirname(__file__), "tenet_plantvillage_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        running_correct = 0
        running_total = 0

        train_iter = train_loader
        if tqdm is not None:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)

        for data, target in train_iter:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            running_correct += (pred == target).sum().item()
            running_total += target.size(0)

            if tqdm is not None:
                train_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{(100.0 * running_correct / max(running_total, 1)):.2f}%"
                )

        train_loss = running_train_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp=use_amp)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

    print(f"Best model saved to: {best_model_path}, best val acc: {best_val_acc:.2f}%")

    # 测试阶段加载最优权重
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, device, use_amp=use_amp)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    train(
        data_root=r"C:\Users\20429\Desktop\project\learn_pytorch\new\PlantVillage-Dataset-master",
        epochs=20,
        batch_size=4,
        lr=1e-3,
        num_layers=4,
        chunk_size=2048,
        image_size=256,
        use_amp=True
    )
