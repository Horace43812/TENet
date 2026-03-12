import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= 1. 连接强度学习器 (Phi) =================
class TrainablePhi(nn.Module):
    def __init__(self):
        super().__init__()
        # 极小规模 MLP，用来决定局部连接的权重
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, vi, vj):
        # 输入 vi, vj 形状为 (M, C)
        combined = torch.stack([vi, vj], dim=-1)  # (M, C, 2)
        return self.net(combined).squeeze(-1)  # (M, C)


# ================= 2. 极致速度代数算子层 =================
class AlgebraicGraphLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 3  # 迹(Trace), 范数(Norm), 均值(Mean)

        # 预设拉普拉斯矩阵的结构
        self.register_buffer('L', self._build_laplacian())
        # 预计算 L 的平方，用于快速求 Frobenius 范数
        self.register_buffer('L_sq', self._build_laplacian() ** 2)

        self.phi = TrainablePhi()

    def _build_laplacian(self):
        """构建 10 顶点的拉普拉斯矩阵结构"""
        A = torch.zeros((10, 10))
        edges = [(0, 4), (1, 4), (0, 5), (2, 5), (0, 6), (3, 6), (1, 7), (2, 7), (1, 8), (3, 8), (2, 9), (3, 9)]
        for i, j in edges: A[i, j] = A[j, i] = 1.0
        return torch.diag(A.sum(dim=1)) - A

    def forward(self, x):
        B, C, H, W = x.shape
        # 2x2 分组并展平
        x_g = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(-1, C, 4)
        v1, v2, v3, v4 = x_g[:, :, 0], x_g[:, :, 1], x_g[:, :, 2], x_g[:, :, 3]

        # 计算 6 个连接点的值
        e12, e13, e14 = self.phi(v1, v2), self.phi(v1, v3), self.phi(v1, v4)
        e23, e24, e34 = self.phi(v2, v3), self.phi(v2, v4), self.phi(v3, v4)

        # 构造顶点向量 z: (M, C, 10)
        z = torch.stack([v1, v2, v3, v4, e12, e13, e14, e23, e24, e34], dim=2)
        z_sq = z ** 2

        # --- 极致速度代数计算 (利用数学公式规避矩阵生成) ---

        # 1. 迹 (Trace): tr(diag(z) L diag(z)) = sum(z_i^2 * L_ii)
        L_diag = torch.diagonal(self.L)  # 获取对角线元素 (10,)
        trace = (z_sq * L_diag).sum(dim=-1)  # 结果形状 (M, C)

        # 2. Frobenius 范数: ||G||_F^2 = sum_i sum_j (z_i * L_ij * z_j)^2
        # 数学等价于: z_sq @ L_sq @ z_sq.T 的对角线元素
        f_norm_sq = torch.matmul(z_sq, self.L_sq) * z_sq
        f_norm = torch.sqrt(f_norm_sq.sum(dim=-1) + 1e-8)  # 结果形状 (M, C)

        # 3. 均值 (Mean): 保留基础灰度信息
        mean_v = (v1 + v2 + v3 + v4) / 4.0  # 结果形状 (M, C)

        # 合并这三个特征 (M, C, 3) -> (M, C*3)
        out = torch.stack([trace, f_norm, mean_v], dim=2).view(-1, self.out_channels)

        # 还原回空间形状 (B, C*3, H//2, W//2)
        return out.view(B, H // 2, W // 2, self.out_channels).permute(0, 3, 1, 2).contiguous()


# ================= 3. 最终 Speed_TENet 架构 =================
class Speed_TENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = AlgebraicGraphLayer(1)  # 1c -> 3c
        self.layer2 = AlgebraicGraphLayer(3)  # 3c -> 9c

        # 注意：这里我们大幅精简了分类器，看看“迹”特征是否足够强
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9 * 7 * 7, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        # 纯代数特征提取不需要 ReLU，因为 Norm 本身已经包含了非线性
        x = self.layer1(x)
        x = self.layer2(x)
        return self.classifier(x)


# 测试代码
if __name__ == "__main__":
    model = Speed_TENet()
    # 打印参数量，你会惊讶于它的精简
    total_params = sum(p.numel() for p in model.parameters())
    print(f"该模型的总参数量为: {total_params}")

    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"输出形状 (应为 [1, 10]): {output.shape}")