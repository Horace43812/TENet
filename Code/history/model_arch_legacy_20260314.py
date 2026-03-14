import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # 输入 vi, vj 形状为 (M, C)
        base = torch.abs(vi - vj)
        combined = torch.stack([vi, vj], dim=-1)  # (M, C, 2)
        logits = self.net(combined).squeeze(-1)  # (M, C)
        temperature = F.softplus(self.temperature_raw) + 1e-4
        residual_weight = F.softplus(self.residual_raw)
        phi_soft = torch.sigmoid(logits / temperature)
        return base + residual_weight * phi_soft


class GraphifiedLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        alpha=1,
        trace_init=0.1,
        phi_temp_init=1.0,
        phi_residual_init=0.1,
        use_learnable_phi=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.alpha = alpha
        self.out_channels = in_channels + self.alpha

        # 预定义拉普拉斯矩阵结构
        self.register_buffer('L', self._build_laplacian())
        self.register_buffer('L_diag', torch.diagonal(self.L))

        # 可学习的 phi
        self.phi = TrainablePhi(phi_temp_init=phi_temp_init, phi_residual_init=phi_residual_init) if use_learnable_phi else None

        # 10x10 图矩阵拓扑提取：使用单层卷积直接映射到输出通道
        self.topology_extractor = nn.Conv2d(in_channels, self.out_channels, kernel_size=10, bias=True)
        self.norm = nn.BatchNorm2d(self.out_channels)
        # trace 补强分支：将每个 2x2 局部块的 trace 特征投影到输出通道
        self.trace_proj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=True)
        main_init = 0.9
        self.main_logit = nn.Parameter(torch.logit(torch.full((self.out_channels,), main_init, dtype=torch.float32)))
        trace_init = float(max(min(trace_init, 1.0 - 1e-4), 1e-4))
        self.trace_logit = nn.Parameter(torch.logit(torch.full((self.out_channels,), trace_init, dtype=torch.float32)))
        edge_sq_init = 0.01
        edge_mul_init = 0.01
        self.edge_sq_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(edge_sq_init, dtype=torch.float32))))
        self.edge_mul_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(edge_mul_init, dtype=torch.float32))))

    def _build_laplacian(self):
        A = torch.zeros((10, 10))
        edges = [(0, 4), (1, 4), (0, 5), (2, 5), (0, 6), (3, 6), (1, 7), (2, 7), (1, 8), (3, 8), (2, 9), (3, 9)]
        for i, j in edges: A[i, j] = A[j, i] = 1.0
        return torch.diag(A.sum(dim=1)) - A

    def forward(self, x):
        B, C, H, W = x.shape
        # 分组 2x2
        x_grouped = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(-1, C, 4)
        v1, v2, v3, v4 = x_grouped[:, :, 0], x_grouped[:, :, 1], x_grouped[:, :, 2], x_grouped[:, :, 3]

        # 计算连接顶点
        if self.phi:
            e12, e13, e14 = self.phi(v1, v2), self.phi(v1, v3), self.phi(v1, v4)
            e23, e24, e34 = self.phi(v2, v3), self.phi(v2, v4), self.phi(v3, v4)
        else:
            e12, e13, e14 = torch.abs(v1 - v2), torch.abs(v1 - v3), torch.abs(v1 - v4)
            e23, e24, e34 = torch.abs(v2 - v3), torch.abs(v2 - v4), torch.abs(v3 - v4)

        d12, d13, d14 = torch.abs(v1 - v2), torch.abs(v1 - v3), torch.abs(v1 - v4)
        d23, d24, d34 = torch.abs(v2 - v3), torch.abs(v2 - v4), torch.abs(v3 - v4)
        p12, p13, p14 = v1 * v2, v1 * v3, v1 * v4
        p23, p24, p34 = v2 * v3, v2 * v4, v3 * v4
        edge_sq_scale = F.softplus(self.edge_sq_raw)
        edge_mul_scale = F.softplus(self.edge_mul_raw)
        e12 = e12 + edge_sq_scale * (d12 ** 2) + edge_mul_scale * p12
        e13 = e13 + edge_sq_scale * (d13 ** 2) + edge_mul_scale * p13
        e14 = e14 + edge_sq_scale * (d14 ** 2) + edge_mul_scale * p14
        e23 = e23 + edge_sq_scale * (d23 ** 2) + edge_mul_scale * p23
        e24 = e24 + edge_sq_scale * (d24 ** 2) + edge_mul_scale * p24
        e34 = e34 + edge_sq_scale * (d34 ** 2) + edge_mul_scale * p34

        z = torch.stack([v1, v2, v3, v4, e12, e13, e14, e23, e24, e34], dim=2)
        z_sq = z ** 2

        # 图化矩阵生成: G = diag(z) * L * diag(z)
        G = z.unsqueeze(3) * self.L * z.unsqueeze(2)
        # 对每个样本/通道上的 10x10 图矩阵做标准化，稳定训练
        G_mean = G.mean(dim=(2, 3), keepdim=True)
        G_std = G.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        G = (G - G_mean) / G_std
        trace = (z_sq * self.L_diag).sum(dim=2)

        # 拓扑特征提取
        main_out = self.topology_extractor(G).squeeze(-1).squeeze(-1)
        main_out = main_out.view(B, H // 2, W // 2, self.out_channels).permute(0, 3, 1, 2).contiguous()
        trace_map = trace.view(B, H // 2, W // 2, C).permute(0, 3, 1, 2).contiguous()
        trace_enhance = self.trace_proj(trace_map)
        main_gate = torch.sigmoid(self.main_logit).view(1, self.out_channels, 1, 1)
        trace_gate = torch.sigmoid(self.trace_logit).view(1, self.out_channels, 1, 1)
        out = main_gate * main_out + trace_gate * trace_enhance

        return F.relu(self.norm(out))


class MNIST_TENet(nn.Module):
    def __init__(
        self,
        alpha=1,
        trace_init=0.1,
        phi_temp_init=1.0,
        phi_residual_init=0.1,
        classifier_dropout=0.2,
        use_learnable_phi=True
    ):
        super().__init__()
        self.alpha = alpha
        self.trace_init = trace_init
        self.phi_temp_init = phi_temp_init
        self.phi_residual_init = phi_residual_init
        self.classifier_dropout = classifier_dropout
        self.use_learnable_phi = use_learnable_phi

        # MNIST 28x28 -> 14x14 -> 7x7
        c1 = 1 + self.alpha
        c2 = c1 + self.alpha
        self.layer1 = GraphifiedLayer(
            1,
            alpha=self.alpha,
            trace_init=self.trace_init,
            phi_temp_init=self.phi_temp_init,
            phi_residual_init=self.phi_residual_init,
            use_learnable_phi=self.use_learnable_phi
        )  # 1->(1+alpha) channels
        self.layer2 = GraphifiedLayer(
            c1,
            alpha=self.alpha,
            trace_init=self.trace_init,
            phi_temp_init=self.phi_temp_init,
            phi_residual_init=self.phi_residual_init,
            use_learnable_phi=self.use_learnable_phi
        )  # (1+alpha)->(1+2alpha) channels
        # 到达 7x7 之后，通道数较少，我们可以增加一个卷积层补强，或者继续压缩
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, 10),
            nn.ReLU(),
            #nn.Dropout(self.classifier_dropout),
            #nn.Linear(48, 10),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.classifier(x)
