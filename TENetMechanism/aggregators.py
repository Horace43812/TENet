"""Controlled single-stream local aggregation; no unused content/trace parameters."""
import torch
from torch import nn
from torch.nn import functional as F

from graph import group_windows, TrainablePhi


OPERATORS = ("P", "V", "Q", "E", "G", "G+", "outer", "mismatch", "Z", "ZQ")


class Aggregation(nn.Module):
    def __init__(self, channels, operator, hidden=32, normalization="standardized",
                 mismatch_seed=0, residual=False):
        super().__init__()
        if operator not in OPERATORS or normalization not in ("standardized", "centered", "raw"):
            raise ValueError("Unknown operator or normalization")
        self.operator, self.normalization = operator, normalization
        self.observer = None
        self.intervention = None
        self.capture_normalization = False
        self.reconstruction_checks = []
        self.use_chebyshev = self.use_trace_branch = self.use_cross_stream_guidance = False
        self.is_graph = operator in ("G", "G+", "outer", "mismatch")
        self.available_stages = ["input", "vertices", "projected", "output"]
        uses_nodes = operator not in ("P", "V", "Q")
        if uses_nodes:
            self.available_stages.append("nodes")
        if self.is_graph:
            self.available_stages += ["graph_raw", "graph_normalized"]
            # Construct the same fixed incidence Laplacian without allocating a graph layer.
            adjacency = torch.zeros(10, 10)
            pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            if operator == "mismatch":
                generator = torch.Generator().manual_seed(mismatch_seed)
                order = torch.randperm(6, generator=generator).tolist()
                if order == list(range(6)):
                    order = order[1:] + order[:1]
                pairs = [pairs[i] for i in order]
            for index, (a, b) in enumerate(pairs, 4):
                adjacency[a, index] = adjacency[index, a] = 1
                adjacency[b, index] = adjacency[index, b] = 1
            self.register_buffer("L", torch.diag(adjacency.sum(1)) - adjacency)
        self.phi = TrainablePhi() if operator == "G+" else None
        if operator == "G+":
            initial = torch.log(torch.expm1(torch.tensor(0.01)))
            self.edge_sq_raw = nn.Parameter(initial.clone())
            self.edge_mul_raw = nn.Parameter(initial.clone())
        factor = 100 if self.is_graph else (10 if uses_nodes else (1 if operator == "P" else 4))
        self.projection = (nn.Sequential(nn.Linear(channels * factor, hidden), nn.ReLU(),
                                        nn.Linear(hidden, channels)) if operator in ("Q", "ZQ")
                           else nn.Linear(channels * factor, channels))
        self.norm = nn.BatchNorm2d(channels)
        self.residual = nn.Conv2d(channels, channels, 1) if residual else None

    def forward(self, x):
        b, c, h, w = x.shape
        def observe(name, value, grouped=False):
            if self.observer:
                if grouped:
                    value = value.reshape(b, h // 2, w // 2, -1).permute(0, 3, 1, 2)
                self.observer(name, value)
        observe("input", x)
        vertices = group_windows(x)
        observe("vertices", vertices, True)
        value = vertices
        if self.operator == "P":
            value = vertices.mean(-1)
        elif self.operator not in ("V", "Q"):
            edges = []
            for a, d in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
                vi, vj = vertices[:, :, a], vertices[:, :, d]
                edge = self.phi(vi, vj) if self.phi is not None else (vi - vj).abs()
                if self.operator == "G+":
                    edge = edge + F.softplus(self.edge_sq_raw) * (vi - vj).square()
                    edge = edge + F.softplus(self.edge_mul_raw) * vi * vj
                edges.append(edge)
            value = torch.cat((vertices, torch.stack(edges, -1)), -1)
            observe("nodes", value, True)
            if self.is_graph:
                raw = value.unsqueeze(-1) * value.unsqueeze(-2)
                if self.operator != "outer":
                    raw = raw * self.L
                observe("graph_raw", raw, True)
                value = raw
                if self.normalization != "raw":
                    mean = raw.mean((-1, -2), keepdim=True)
                    value = raw - mean
                    if self.normalization == "standardized":
                        scale = raw.std((-1, -2), keepdim=True).clamp_min(1e-6)
                        standardized = value / scale
                        value = standardized
                if self.intervention == "no_graph_norm":
                    value = raw
                observe("graph_normalized", value, True)
                if self.capture_normalization:
                    if self.normalization != "standardized" or self.intervention:
                        raise ValueError("A-E study requires standardized baseline")
                    restored = standardized * scale + mean
                    passed = torch.allclose(raw, restored, rtol=1e-5, atol=1e-5)
                    self.reconstruction_checks.append({"passed": passed,
                        "max_abs_error": float((raw - restored).abs().max().detach()),
                        "max_abs_raw": float(raw.abs().max().detach())})
                    if not passed:
                        raise RuntimeError("A/E reconstruction failed")
                    observe("graph_scale_restored", standardized * scale, True)
                    observe("graph_mean_restored", standardized + mean, True)
                    observe("graph_restored", restored, True)
            elif self.operator in ("Z", "ZQ"):
                value = (value - value.mean(-1, keepdim=True)) / value.std(-1, keepdim=True).clamp_min(1e-6)
        output = self.projection(value.flatten(1))
        output = output.reshape(b, h // 2, w // 2, c).permute(0, 3, 1, 2).contiguous()
        observe("projected", output)
        if self.residual is not None:
            output = output + self.residual(F.avg_pool2d(x, 2))
        output = F.relu(self.norm(output))
        observe("output", output)
        return output


class ResearchNet(nn.Module):
    def __init__(self, num_classes, image_size=64, channels=16, operators=("G", "G", "G"),
                 hidden=32, normalization="standardized", head_grid=4, head_hidden=256,
                 dropout=0.1, mismatch_seed=0, residual=False, schedule=None):
        super().__init__()
        if min(num_classes, image_size, channels, hidden, head_grid, head_hidden) < 1 or not operators:
            raise ValueError("Invalid model sizes")
        if image_size % 2 ** len(operators) or image_size // 2 ** len(operators) < head_grid:
            raise ValueError("Incompatible input, aggregation depth and classifier grid")
        self.stem = nn.Identity() if channels == 3 else nn.Conv2d(3, channels, 1)
        self.layers = nn.ModuleList([Aggregation(channels, op, hidden, normalization, mismatch_seed, residual)
                                     for op in operators])
        self.schedule = schedule or "D" * len(operators)
        if self.schedule.count("D") != len(operators) or set(self.schedule) - {"D", "P"}:
            raise ValueError("Schedule must have one D per aggregation stage")
        self.processing = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU()) for _ in range(self.schedule.count("P"))])
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(head_grid), nn.Flatten(),
            nn.Linear(channels * head_grid ** 2, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes))

    def forward(self, x):
        x = self.stem(x)
        d = p = 0
        for block in self.schedule:
            if block == "D":
                x = self.layers[d](x)
                d += 1
            else:
                x = self.processing[p](x)
                p += 1
        return self.classifier(x)
