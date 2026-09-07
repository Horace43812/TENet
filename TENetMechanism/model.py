"""Standalone recursive TENet, adapted from Code/model_common.py and model_rgb32.py."""

import torch
from torch import nn
from graph import GraphifiedLayer


def build_model(config):
    """Old configurations retain their exact architecture; new ones are explicit."""
    config = dict(config)
    architecture = config.pop("architecture", "tenet")
    if architecture == "tenet":
        return TENet(**config)
    if architecture == "research":
        from aggregators import ResearchNet
        return ResearchNet(**config)
    raise ValueError(f"Unknown checkpoint architecture: {architecture}")


class TENet(nn.Module):
    def __init__(self, num_classes, image_size=64, channels=3, layers=3, alpha=8,
                 head_grid=4, head_hidden=256, dropout=0.1, readout="topology",
                 learnable_phi=True, trace=True, edge=True, directed=False, cross=True,
                 chebyshev=False, spatial=False):
        super().__init__()
        if min(image_size, channels, layers, head_grid, head_hidden) < 1 or alpha < 0:
            raise ValueError("Model sizes must be positive; alpha must be nonnegative")
        if image_size % (2 ** layers) or head_grid > image_size // (2 ** layers):
            raise ValueError("image_size must divide by 2**layers and final grid must be >= head_grid")
        if readout not in ("topology", "content", "both"):
            raise ValueError("Unknown readout")
        self.readout = readout
        self.stem = nn.Identity() if channels == 3 else nn.Conv2d(3, channels, 1)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(GraphifiedLayer(channels, alpha=alpha,
                use_learnable_phi=learnable_phi, use_trace_branch=trace, use_edge_enhance=edge,
                use_directed_flow=directed, use_cross_stream_guidance=cross,
                use_chebyshev=chebyshev, use_spatial_bridge=spatial))
            channels += alpha
        width = channels * (2 if readout == "both" else 1)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(head_grid), nn.Flatten(),
            nn.Linear(width * head_grid ** 2, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes))

    def forward(self, images):
        topo = content = self.stem(images)
        for layer in self.layers:
            topo, content = layer(topo, content)
        features = {"topology": topo, "content": content}
        if self.readout == "both":
            features["both"] = torch.cat((topo, content), dim=1)
        return self.classifier(features[self.readout])
