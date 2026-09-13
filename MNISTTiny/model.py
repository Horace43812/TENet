"""Two graph stages (1 -> 2 -> 3 channels), 2,295 trainable parameters."""
import torch
from torch import nn


class TinyGraph(nn.Module):
    def __init__(self, in_channels, out_channels, normalization="raw"):
        super().__init__()
        if normalization not in ("raw", "centered", "standardized"):
            raise ValueError("Unknown graph normalization")
        self.normalization = normalization
        self.pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        adjacency = torch.zeros(10, 10)
        for edge, (a, b) in enumerate(self.pairs, 4):
            adjacency[a, edge] = adjacency[edge, a] = 1
            adjacency[b, edge] = adjacency[edge, b] = 1
        self.register_buffer("L", torch.diag(adjacency.sum(1)) - adjacency)
        self.projection = nn.Linear(in_channels * 100, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)

    def graph_matrix(self, x):
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError("Graph stage requires even spatial dimensions")
        vertices = x.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(-1, c, 4)
        edges = torch.stack([(vertices[..., a] - vertices[..., d]).abs() for a, d in self.pairs], -1)
        z = torch.cat((vertices, edges), -1)
        raw = z.unsqueeze(-1) * self.L * z.unsqueeze(-2)
        if self.normalization == "raw":
            return raw
        centered = raw - raw.mean((-1, -2), keepdim=True)
        if self.normalization == "centered":
            return centered
        return centered / raw.std((-1, -2), keepdim=True).clamp_min(1e-6)

    def forward(self, x):
        b, _, h, w = x.shape
        output = self.projection(self.graph_matrix(x).flatten(1))
        output = output.reshape(b, h // 2, w // 2, -1).permute(0, 3, 1, 2).contiguous()
        return torch.relu(self.bn(output))


class TinyMNIST(nn.Module):
    def __init__(self, normalization="raw"):
        super().__init__()
        self.layer1 = TinyGraph(1, 2, normalization)
        self.layer2 = TinyGraph(2, 3, normalization)
        self.classifier = nn.Linear(3 * 7 * 7, 10)

    def forward(self, x):
        if x.shape[1:] != (1, 28, 28):
            raise ValueError("Expected B x 1 x 28 x 28 MNIST images")
        return self.classifier(self.layer2(self.layer1(x)).flatten(1))
