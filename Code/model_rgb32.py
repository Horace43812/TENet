import torch
import torch.nn as nn

from model_common import GraphifiedLayer


class TENet_RGB32(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_layers=2,
        input_size=32,
        alpha=2,
        trace_init=0.1,
        phi_temp_init=1.0,
        phi_residual_init=0.1,
        classifier_dropout=0.1,
        use_learnable_phi=True,
        use_trace_branch=True,
        use_edge_enhance=True,
        use_directed_flow=False,
        use_cross_stream_guidance=True,
        use_chebyshev=False,
        use_spatial_bridge=False
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1")
        if input_size % (2 ** num_layers) != 0:
            raise ValueError(
                f"input_size={input_size} is not divisible by 2^{num_layers}; "
                f"final spatial size must be integer."
            )

        self.num_layers = num_layers
        self.final_spatial = input_size // (2 ** num_layers)

        layers = []
        topo_channels = 3
        content_channels = 3
        for _ in range(num_layers):
            layer = GraphifiedLayer(
                in_channels=topo_channels,
                content_in_channels=content_channels,
                alpha=alpha,
                trace_init=trace_init,
                phi_temp_init=phi_temp_init,
                phi_residual_init=phi_residual_init,
                use_learnable_phi=use_learnable_phi,
                use_trace_branch=use_trace_branch,
                use_edge_enhance=use_edge_enhance,
                use_directed_flow=use_directed_flow,
                use_cross_stream_guidance=use_cross_stream_guidance,
                use_chebyshev=use_chebyshev,
                use_spatial_bridge=use_spatial_bridge
            )
            layers.append(layer)
            topo_channels = layer.out_channels
            content_channels = layer.content_out_channels

        self.layers = nn.ModuleList(layers)
        self.final_channels = topo_channels

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_channels * self.final_spatial * self.final_spatial, 192),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x_topo = x
        x_content = x
        for layer in self.layers:
            x_topo, x_content = layer(x_topo, x_content)
        return self.classifier(x_topo)
