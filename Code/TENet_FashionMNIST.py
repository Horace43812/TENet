import torch
import torch.nn as nn
import torch.optim as optim
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
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 新增：修复下载链接的代码 ---
    # 强制指定国内或更稳定的镜像源（这里使用常用的镜像规则）
    datasets.FashionMNIST.resources = [
        ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
         "f68b3c2dcbeaaa9fbdd348bbdeb45c12"),
        ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
         "f17e45b128099976906a5758510dd100"),
        ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
         "bef4ecab320f06d8554227c12da6a029"),
        ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
         "bb300cfdad3c16e7a124157e65021045")
    ]
    # ----------------------------

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

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
