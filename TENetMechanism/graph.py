import torch
import torch.nn as nn
import torch.nn.functional as F


def group_windows(x):
    """B,C,H,W -> (B*H/2*W/2),C,4, raster ordered within each window."""
    b, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError("Graph aggregation requires even spatial dimensions")
    return x.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(-1, c, 4)


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
        combined = torch.stack([vi, vj], dim=-1)
        logits = self.net(combined).squeeze(-1)
        temperature = F.softplus(self.temperature_raw) + 1e-4
        residual_weight = F.softplus(self.residual_raw)
        phi_soft = torch.sigmoid(logits / temperature)
        return base + residual_weight * phi_soft


class GraphifiedLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        content_in_channels=None,
        alpha=1,
        trace_init=0.1,
        phi_temp_init=1.0,
        phi_residual_init=0.1,
        use_learnable_phi=True,
        use_trace_branch=True,
        use_edge_enhance=True,
        use_directed_flow=True,
        use_cross_stream_guidance=True,
        use_chebyshev=False,
        use_spatial_bridge=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.content_in_channels = in_channels if content_in_channels is None else content_in_channels
        self.alpha = alpha
        self.out_channels = in_channels + self.alpha
        self.content_out_channels = self.content_in_channels + self.alpha
        self.use_trace_branch = use_trace_branch
        self.use_edge_enhance = use_edge_enhance
        self.use_directed_flow = use_directed_flow
        self.use_cross_stream_guidance = use_cross_stream_guidance
        self.use_chebyshev = use_chebyshev
        self.use_spatial_bridge = use_spatial_bridge
        self.observer = None
        self.intervention = None

        self.register_buffer('L', self._build_laplacian())
        if self.use_directed_flow:
            self.register_buffer('B_incidence', self._build_incidence())
        if self.use_chebyshev:
            self.register_buffer('L2', self.L @ self.L)
        self.register_buffer('L_diag', torch.diagonal(self.L))

        self.phi = TrainablePhi(phi_temp_init=phi_temp_init, phi_residual_init=phi_residual_init) if use_learnable_phi else None

        self.topology_extractor = nn.Conv2d(in_channels, self.out_channels, kernel_size=10, bias=True)
        if self.use_chebyshev:
            self.topo_mix_logits = nn.Parameter(torch.tensor([2.2, -2.2], dtype=torch.float32))
        self.norm = nn.BatchNorm2d(self.out_channels)
        if self.use_spatial_bridge:
            self.spatial_dw = nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=self.out_channels,
                bias=False
            )
            self.spatial_gate = nn.Parameter(torch.full((self.out_channels,), 0.01, dtype=torch.float32))
        self.content_proj = nn.Conv2d(self.content_in_channels, self.content_out_channels, kernel_size=1, bias=True)
        self.content_norm = nn.BatchNorm2d(self.content_out_channels)
        if self.use_cross_stream_guidance:
            self.align_proj = nn.Conv2d(self.content_out_channels, self.out_channels, kernel_size=1, bias=True)
            self.topo_to_content_proj = nn.Conv2d(self.out_channels, self.content_out_channels, kernel_size=1, bias=True)
            self.gamma = nn.Parameter(torch.full((self.out_channels,), 0.01, dtype=torch.float32))
            self.beta = nn.Parameter(torch.full((self.content_out_channels,), 0.01, dtype=torch.float32))
        if self.use_trace_branch:
            self.trace_proj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=True)
        main_init = 0.9
        self.main_logit = nn.Parameter(torch.logit(torch.full((self.out_channels,), main_init, dtype=torch.float32)))
        if self.use_trace_branch:
            trace_init = float(max(min(trace_init, 1.0 - 1e-4), 1e-4))
            self.trace_logit = nn.Parameter(torch.logit(torch.full((self.out_channels,), trace_init, dtype=torch.float32)))
        edge_sq_init = 0.01
        edge_mul_init = 0.01
        if self.use_edge_enhance:
            self.edge_sq_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(edge_sq_init, dtype=torch.float32))))
            self.edge_mul_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(edge_mul_init, dtype=torch.float32))))
        if self.use_directed_flow:
            self.flow_gate = nn.Parameter(torch.full((6,), 0.01, dtype=torch.float32))

    def _build_laplacian(self):
        A = torch.zeros((10, 10))
        edges = [(0, 4), (1, 4), (0, 5), (2, 5), (0, 6), (3, 6), (1, 7), (2, 7), (1, 8), (3, 8), (2, 9), (3, 9)]
        for i, j in edges:
            A[i, j] = A[j, i] = 1.0
        return torch.diag(A.sum(dim=1)) - A

    def _build_incidence(self):
        B = torch.zeros((4, 6))
        oriented_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for edge_idx, (src, dst) in enumerate(oriented_edges):
            B[src, edge_idx] = 1.0
            B[dst, edge_idx] = -1.0
        return B

    def forward(self, x_topo, x_content=None):
        if x_content is None:
            x_content = x_topo

        batch_size, channels, height, width = x_topo.shape
        _, content_channels, content_h, content_w = x_content.shape
        if content_h != height or content_w != width:
            raise ValueError(f"Spatial size mismatch between topo/content streams: topo=({height}, {width}), content=({content_h}, {content_w})")

        def observe(name, value, grouped=False):
            if self.observer is not None:
                if grouped:
                    value = value.reshape(batch_size, height // 2, width // 2, -1).permute(0, 3, 1, 2)
                self.observer(name, value)

        observe("input", x_topo)
        observe("content_input", x_content)
        x_grouped = group_windows(x_topo)
        observe("vertices", x_grouped, True)
        v1, v2, v3, v4 = x_grouped[:, :, 0], x_grouped[:, :, 1], x_grouped[:, :, 2], x_grouped[:, :, 3]
        if self.use_directed_flow:
            vertex_stack = torch.stack([v1, v2, v3, v4], dim=2)
            flow = torch.matmul(vertex_stack, self.B_incidence)
            flow12, flow13, flow14, flow23, flow24, flow34 = flow.unbind(dim=2)

        x_content_grouped = group_windows(x_content)
        v_mean = x_content_grouped.mean(dim=-1)
        v_mean = v_mean.view(batch_size, height // 2, width // 2, content_channels).permute(0, 3, 1, 2).contiguous()
        x_content_out = F.relu(self.content_norm(self.content_proj(v_mean)))

        if self.phi:
            e12, e13, e14 = self.phi(v1, v2), self.phi(v1, v3), self.phi(v1, v4)
            e23, e24, e34 = self.phi(v2, v3), self.phi(v2, v4), self.phi(v3, v4)
        else:
            e12, e13, e14 = torch.abs(v1 - v2), torch.abs(v1 - v3), torch.abs(v1 - v4)
            e23, e24, e34 = torch.abs(v2 - v3), torch.abs(v2 - v4), torch.abs(v3 - v4)

        if self.use_edge_enhance:
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

        if self.use_directed_flow:
            flow_weight = torch.tanh(self.flow_gate)
            e12 = e12 + flow_weight[0] * flow12
            e13 = e13 + flow_weight[1] * flow13
            e14 = e14 + flow_weight[2] * flow14
            e23 = e23 + flow_weight[3] * flow23
            e24 = e24 + flow_weight[4] * flow24
            e34 = e34 + flow_weight[5] * flow34

        z = torch.stack([v1, v2, v3, v4, e12, e13, e14, e23, e24, e34], dim=2)
        observe("nodes", z, True)
        z_sq = z ** 2
        raw_graph = z.unsqueeze(3) * self.L * z.unsqueeze(2)
        observe("graph_raw", raw_graph, True)

        if self.use_chebyshev:
            g1 = z.unsqueeze(3) * self.L * z.unsqueeze(2)
            g2 = z.unsqueeze(3) * self.L2 * z.unsqueeze(2)
            g1 = (g1 - g1.mean(dim=(2, 3), keepdim=True)) / g1.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            g2 = (g2 - g2.mean(dim=(2, 3), keepdim=True)) / g2.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            topo_mix = torch.softmax(self.topo_mix_logits, dim=0)
            g = topo_mix[0] * g1 + topo_mix[1] * g2
        else:
            g = raw_graph
            g_mean = g.mean(dim=(2, 3), keepdim=True)
            g_std = g.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            g = (g - g_mean) / g_std
        if self.intervention == "no_graph_norm":
            g = raw_graph
        observe("graph_normalized", g, True)

        trace = (z_sq * self.L_diag).sum(dim=2)
        observe("trace", trace, True)

        main_out = self.topology_extractor(g).squeeze(-1).squeeze(-1)
        main_out = main_out.view(batch_size, height // 2, width // 2, self.out_channels).permute(0, 3, 1, 2).contiguous()
        observe("projected", main_out)
        if self.use_trace_branch:
            trace_map = trace.view(batch_size, height // 2, width // 2, channels).permute(0, 3, 1, 2).contiguous()
            trace_enhance = self.trace_proj(trace_map)
            if self.intervention == "no_trace":
                trace_enhance = torch.zeros_like(trace_enhance)
            main_gate = torch.sigmoid(self.main_logit).view(1, self.out_channels, 1, 1)
            trace_gate = torch.sigmoid(self.trace_logit).view(1, self.out_channels, 1, 1)
            x_topo_out = main_gate * main_out + trace_gate * trace_enhance
        else:
            x_topo_out = main_out

        if self.use_cross_stream_guidance and self.intervention != "no_cross":
            c_to_t_guidance = self.align_proj(x_content_out)
            gamma_weighted = torch.tanh(self.gamma).view(1, self.out_channels, 1, 1)
            x_topo_out = x_topo_out + gamma_weighted * c_to_t_guidance

        observe("pre_activation", x_topo_out)
        x_topo_out = F.relu(self.norm(x_topo_out))
        if self.use_spatial_bridge:
            spatial_guidance = self.spatial_dw(x_topo_out)
            spatial_gate = torch.tanh(self.spatial_gate).view(1, self.out_channels, 1, 1)
            x_topo_out = x_topo_out + spatial_gate * spatial_guidance

        if self.use_cross_stream_guidance and self.intervention != "no_cross":
            t_to_c_guidance = self.topo_to_content_proj(x_topo_out)
            beta_weighted = torch.tanh(self.beta).view(1, self.content_out_channels, 1, 1)
            x_content_out = x_content_out + beta_weighted * t_to_c_guidance

        x_topo_out = F.relu(x_topo_out)
        x_content_out = F.relu(x_content_out)
        observe("output", x_topo_out)
        observe("content_output", x_content_out)
        return x_topo_out, x_content_out
