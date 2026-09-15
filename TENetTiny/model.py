"""Historical current/legacy architectures with explicit experimental controls."""
import torch
from torch import nn
from torch.nn import functional as F
from tiny_v1 import TinyGraph, TinyMNIST  # Original 2,295-parameter checkpoint compatibility.


class TrainablePhi(nn.Module):
    def __init__(self, temperature=1., residual=.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
        self.temperature_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(max(float(temperature), 1e-4)))))
        self.residual_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(max(float(residual), 1e-6)))))

    def forward(self, a, b):
        logits = self.net(torch.stack([a, b], -1)).squeeze(-1)
        return (a - b).abs() + F.softplus(self.residual_raw) * torch.sigmoid(logits / (F.softplus(self.temperature_raw) + 1e-4))


class GraphLayer(nn.Module):
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

    def __init__(self, channels, alpha, *, content, learnable_phi, trace, edge, directed,
                 cross, chebyshev, spatial, main_gate, normalization, window_layout,
                 trace_init, phi_temp_init, phi_residual_init):
        super().__init__()
        self.out_channels = channels + alpha
        out = self.out_channels
        self.content, self.trace, self.edge = content, trace, edge
        self.directed, self.cross, self.chebyshev, self.spatial = directed, cross, chebyshev, spatial
        self.main_gate, self.normalization, self.window_layout = main_gate, normalization, window_layout
        adjacency = torch.zeros(10, 10)
        incidence = torch.zeros(4, 6)
        for index, (a, b) in enumerate(self.pairs):
            adjacency[a, index + 4] = adjacency[index + 4, a] = 1
            adjacency[b, index + 4] = adjacency[index + 4, b] = 1
            incidence[a, index], incidence[b, index] = 1, -1
        self.register_buffer('L', torch.diag(adjacency.sum(1)) - adjacency)
        if directed:
            self.register_buffer('B_incidence', incidence)
        if chebyshev:
            self.register_buffer('L2', self.L @ self.L)
        self.register_buffer('L_diag', torch.diagonal(self.L))
        self.phi = TrainablePhi(phi_temp_init, phi_residual_init) if learnable_phi else None
        self.topology_extractor = nn.Conv2d(channels, out, 10, bias=True)
        if chebyshev:
            self.topo_mix_logits = nn.Parameter(torch.tensor([2.2, -2.2]))
        self.norm = nn.BatchNorm2d(out)
        if spatial:
            self.spatial_dw = nn.Conv2d(out, out, 3, padding=2, dilation=2, groups=out, bias=False)
            self.spatial_gate = nn.Parameter(torch.full((out,), .01))
        if content:
            self.content_proj = nn.Conv2d(channels, out, 1)
            self.content_norm = nn.BatchNorm2d(out)
        if cross:
            self.align_proj = nn.Conv2d(out, out, 1)
            self.topo_to_content_proj = nn.Conv2d(out, out, 1)
            self.gamma = nn.Parameter(torch.full((out,), .01))
            self.beta = nn.Parameter(torch.full((out,), .01))
        if trace:
            self.trace_proj = nn.Conv2d(channels, out, 1)
        if main_gate:
            self.main_logit = nn.Parameter(torch.logit(torch.full((out,), .9)))
        if trace:
            self.trace_logit = nn.Parameter(torch.logit(torch.full((out,), max(min(float(trace_init), 1 - 1e-4), 1e-4))))
        if edge:
            self.edge_sq_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(.01))))
            self.edge_mul_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(.01))))
        if directed:
            self.flow_gate = nn.Parameter(torch.full((6,), .01))

    def windows(self, x):
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError('Graph layers require even H/W')
        x = x.reshape(b, c, h // 2, 2, w // 2, 2)
        order = (0, 1, 2, 4, 3, 5) if self.window_layout == 'historical' else (0, 2, 4, 1, 3, 5)
        return x.permute(*order).reshape(-1, c, 4)

    def normalize_graph(self, graph):
        if self.normalization == 'raw':
            return graph
        centered = graph - graph.mean((2, 3), keepdim=True)
        return centered if self.normalization == 'centered' else centered / graph.std((2, 3), keepdim=True).clamp_min(1e-6)

    def forward(self, x, content=None):
        b, c, h, w = x.shape
        def spatial_map(value):
            return value.reshape(b, h // 2, w // 2, -1).permute(0, 3, 1, 2).contiguous()
        v = self.windows(x)
        content_out = None
        if self.content:
            pooled = spatial_map(self.windows(x if content is None else content).mean(-1))
            content_out = F.relu(self.content_norm(self.content_proj(pooled)))
        flow = v @ self.B_incidence if self.directed else None
        edges = []
        for index, (a, d) in enumerate(self.pairs):
            va, vd = v[:, :, a], v[:, :, d]
            value = self.phi(va, vd) if self.phi is not None else (va - vd).abs()
            if self.edge:
                value = value + F.softplus(self.edge_sq_raw) * (va - vd).abs().square() + F.softplus(self.edge_mul_raw) * (va * vd)
            if self.directed:
                value = value + torch.tanh(self.flow_gate)[index] * flow[:, :, index]
            edges.append(value)
        z = torch.cat((v, torch.stack(edges, -1)), -1)
        graph = self.normalize_graph(z.unsqueeze(3) * self.L * z.unsqueeze(2))
        if self.chebyshev:
            second = self.normalize_graph(z.unsqueeze(3) * self.L2 * z.unsqueeze(2))
            mix = self.topo_mix_logits.softmax(0)
            graph = mix[0] * graph + mix[1] * second
        output = spatial_map(self.topology_extractor(graph).squeeze(-1).squeeze(-1))
        if self.trace:
            trace = spatial_map((z.square() * self.L_diag).sum(-1))
            if self.main_gate:
                output = self.main_logit.sigmoid().view(1, -1, 1, 1) * output
            output = output + self.trace_logit.sigmoid().view(1, -1, 1, 1) * self.trace_proj(trace)
        # Historical current bypassed main_logit when trace was disabled.
        if self.cross:
            output = output + self.gamma.tanh().view(1, -1, 1, 1) * self.align_proj(content_out)
        output = F.relu(self.norm(output))
        if self.spatial:
            output = output + self.spatial_gate.tanh().view(1, -1, 1, 1) * self.spatial_dw(output)
        if self.cross:
            content_out = content_out + self.beta.tanh().view(1, -1, 1, 1) * self.topo_to_content_proj(output)
        return F.relu(output), F.relu(content_out) if content_out is not None else None


DEFAULTS = {
    'legacy': dict(learnable_phi=True, trace=True, edge=True, directed=False, cross=False,
                   chebyshev=False, spatial=False, content=False, main_gate=True,
                   normalization='standardized', window_layout='historical', head='linear-relu', classifier_dropout=.2),
    'current': dict(learnable_phi=True, trace=True, edge=True, directed=True, cross=True,
                    chebyshev=False, spatial=False, content=True, main_gate=True,
                    normalization='standardized', window_layout='historical', head='mlp', classifier_dropout=.1),
    'tiny': dict(learnable_phi=False, trace=False, edge=False, directed=False, cross=False,
                 chebyshev=False, spatial=False, content=False, main_gate=False,
                 normalization='raw', window_layout='corrected', head='linear', classifier_dropout=0.),
}


class TENetTiny(nn.Module):
    def __init__(self, model_version='current', in_channels=1, num_classes=10, image_size=28,
                 layers=2, alpha=1, head_hidden=48, trace_init=.1, phi_temp_init=1.,
                 phi_residual_init=.1, **overrides):
        super().__init__()
        if model_version not in DEFAULTS:
            raise ValueError('Unknown model version')
        opts = {**DEFAULTS[model_version], **{k: v for k, v in overrides.items() if v is not None}}
        if set(opts) != set(DEFAULTS[model_version]):
            raise ValueError('Unknown model option')
        if min(in_channels, num_classes, image_size, layers, head_hidden) < 1 or alpha < 0:
            raise ValueError('Invalid model dimensions')
        if image_size % (2 ** layers):
            raise ValueError('image-size must be divisible by 2**layers')
        if opts['normalization'] not in ('raw', 'centered', 'standardized') or opts['window_layout'] not in ('historical', 'corrected'):
            raise ValueError('Invalid normalization or window layout')
        if opts['cross']:
            if overrides.get('content') is False:
                raise ValueError('cross requires content; use --content or --no-cross')
            opts['content'] = True
        if opts['head'] not in ('linear', 'linear-relu', 'mlp') or not 0 <= opts['classifier_dropout'] < 1:
            raise ValueError('Invalid classifier options')
        self.resolved_options = opts.copy()
        self.layers_count, self.image_size, self.in_channels = layers, image_size, in_channels
        head = opts.pop('head')
        dropout = opts.pop('classifier_dropout')
        for index in range(layers):
            setattr(self, f'layer{index + 1}', GraphLayer(in_channels + index * alpha, alpha,
                trace_init=trace_init, phi_temp_init=phi_temp_init, phi_residual_init=phi_residual_init, **opts))
        features = (in_channels + layers * alpha) * (image_size // 2 ** layers) ** 2
        modules = [nn.Flatten()]
        if head == 'mlp':
            modules += [nn.Linear(features, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, num_classes)]
        else:
            modules += [nn.Linear(features, num_classes)]
            if head == 'linear-relu':
                modules += [nn.ReLU()]
        self.classifier = nn.Sequential(*modules)

    def forward(self, x):
        if x.shape[1:] != (self.in_channels, self.image_size, self.image_size):
            raise ValueError('Unexpected input shape')
        content = x
        for index in range(self.layers_count):
            x, content = getattr(self, f'layer{index + 1}')(x, content)
        return self.classifier(x)
