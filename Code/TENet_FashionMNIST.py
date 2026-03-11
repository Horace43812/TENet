import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# 1. TENet 核心算子：图化演化层
# ==========================================
class AlgebraicGraphLayer(nn.Module):
    def __init__(self, in_features):
        super(AlgebraicGraphLayer, self).__init__()
        # Phi 网络：根据像素值学习连接强度 e (6个连接对应2x2邻域)
        self.phi = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
            nn.Sigmoid()
        )

        # 预定义 10x10 的拉普拉斯骨架 L (简化索引映射)
        # 这里为了演示，我们直接在 forward 中构建动态 G
        self.register_buffer('L_mask', self._create_laplacian_mask())

    def _create_laplacian_mask(self):
        # 这是一个 10x10 的拓扑模板，定义了 4个顶点和 6个边的连接关系
        L = torch.zeros(10, 10)
        # 简化版：仅定义顶点与边的交互位置
        connections = [(0, 4), (1, 4), (0, 5), (2, 5), (0, 6), (3, 6), (1, 7), (2, 7), (1, 8), (3, 8), (2, 9), (3, 9)]
        for v, e in connections:
            L[v, e] = L[e, v] = -1
        for i in range(10): L[i, i] = 3 if i < 4 else 2
        return L

    def forward(self, x):
        # x: [batch, 4] (2x2 patch)
        e = self.phi(x)  # [batch, 6]
        z = torch.cat([x, e], dim=1)  # [batch, 10] (顶点+边)

        # 模拟 G = diag(z) L diag(z) 的迹 (Trace) 提取特征
        # 这是一个极简的代数演化特征：z_i * L_ii * z_i + sum(z_i * L_ij * z_j)
        # 这里我们直接输出演化后的特征向量
        feat = z * (z @ self.L_mask)
        return feat  # 返回 10 维图特征


# ==========================================
# 2. TENet 整体架构 (针对 Fashion-MNIST)
# ==========================================
class TENet_Fashion(nn.Module):
    def __init__(self):
        super(TENet_Fashion, self).__init__()
        self.layer1 = AlgebraicGraphLayer(4)  # 处理 2x2
        # 28x28 -> 14x14 (每个 2x2 产出 1个特征，这里简化为取迹的均值)

        # 分类器：你可以根据你的 48 维实验调整
        # 假设经过两层演化后进入全连接
        self.classifier = nn.Sequential(
            nn.Linear(14 * 14, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 10)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # 1. 空间切分 2x2 patches
        x = x.view(batch_size, 1, 28, 28)
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # [B, 1, 14, 14, 2, 2]
        patches = patches.contiguous().view(-1, 4)

        # 2. 图化演化
        g_feat = self.layer1(patches)  # [-1, 10]
        # 3. 特征压缩 (取 Trace 的物理含义：图能量)
        energy = g_feat.sum(dim=1).view(batch_size, 14 * 14)

        # 4. 分类
        return self.classifier(energy)


# ==========================================
# 3. 训练脚本
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fashion_mnist_resource_variants():
    github_raw = [
        ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
         "f68b3c2dcbeaaa9fbdd348bbdeb45c12"),
        ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
         "f17e45b128099976906a5758510dd100"),
        ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
         "bef4ecab320f06d8554227c12da6a029"),
        ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
         "bb300cfdad3c16e7a124157e65021045")
    ]

    github_content = [
        ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",
         "f68b3c2dcbeaaa9fbdd348bbdeb45c12"),
        ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz",
         "f17e45b128099976906a5758510dd100"),
        ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",
         "bef4ecab320f06d8554227c12da6a029"),
        ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz",
         "bb300cfdad3c16e7a124157e65021045")
    ]

    jsdelivr = [
        ("https://cdn.jsdelivr.net/gh/zalandoresearch/fashion-mnist@master/data/fashion/train-images-idx3-ubyte.gz",
         "f68b3c2dcbeaaa9fbdd348bbdeb45c12"),
        ("https://cdn.jsdelivr.net/gh/zalandoresearch/fashion-mnist@master/data/fashion/train-labels-idx1-ubyte.gz",
         "f17e45b128099976906a5758510dd100"),
        ("https://cdn.jsdelivr.net/gh/zalandoresearch/fashion-mnist@master/data/fashion/t10k-images-idx3-ubyte.gz",
         "bef4ecab320f06d8554227c12da6a029"),
        ("https://cdn.jsdelivr.net/gh/zalandoresearch/fashion-mnist@master/data/fashion/t10k-labels-idx1-ubyte.gz",
         "bb300cfdad3c16e7a124157e65021045")
    ]

    return [github_raw, github_content, jsdelivr]


def _load_fashion_mnist_with_fallback(data_root, train, transform):
    errors = []
    for idx, resources in enumerate(_fashion_mnist_resource_variants(), start=1):
        datasets.FashionMNIST.resources = resources
        try:
            split = "train" if train else "test"
            print(f"Trying FashionMNIST mirror #{idx} for {split}...")
            return datasets.FashionMNIST(root=data_root, train=train, download=True, transform=transform)
        except Exception as exc:
            errors.append(f"mirror #{idx}: {exc}")

    # 所有下载源失败时，尝试直接读取本地缓存
    try:
        split = "train" if train else "test"
        print(f"All mirrors failed, trying local cached {split} set...")
        return datasets.FashionMNIST(root=data_root, train=train, download=False, transform=transform)
    except Exception as exc:
        details = "\n".join(errors)
        raise RuntimeError(
            f"FashionMNIST download failed for all mirrors. Root: {data_root}\n{details}\nlocal-cache: {exc}"
        )


def build_dataloaders(data_root=None, batch_size=64, test_batch_size=1000, seed=42):
    if data_root is None:
        data_root = './data'
    os.makedirs(data_root, exist_ok=True)

    # FashionMNIST 常用统计量
    mean, std = (0.2860,), (0.3530,)
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = _load_fashion_mnist_with_fallback(data_root, train=True, transform=train_transform)
    test_set = _load_fashion_mnist_with_fallback(data_root, train=False, transform=test_transform)

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=loader_generator,
        drop_last=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    print(f"Dataset ready: train={len(train_set)}, test={len(test_set)}, workers={num_workers}")
    return train_loader, test_loader


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(batch_size=64, test_batch_size=1000, seed=42)

    model = TENet_Fashion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(1, 11):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # 测试
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"Epoch {epoch}: Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    train()
