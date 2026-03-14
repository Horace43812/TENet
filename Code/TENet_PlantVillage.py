import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model_rgb256 import TENet_RGB256

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


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

    model = TENet_RGB256(
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
    best_model_path = os.path.join(os.path.dirname(__file__), "tenet_rgb256_best.pth")

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
