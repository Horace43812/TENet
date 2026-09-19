"""Text classifiers sharing token embeddings and masked mean pooling."""
import torch
from torch import nn
from torch.nn import functional as F


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, model_version='tenet', embedding_dim=16,
                 hidden_dim=16, normalization='raw', dropout=0.):
        super().__init__()
        if model_version not in ('tenet', 'cnn', 'mean'):
            raise ValueError('Unknown text model')
        if normalization not in ('raw', 'centered', 'standardized'):
            raise ValueError('Unknown graph normalization')
        self.version, self.normalization = model_version, normalization
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if model_version == 'tenet':
            adjacency = torch.zeros(10, 10)
            pairs = torch.combinations(torch.arange(4), r=2)
            for i, (a, b) in enumerate(pairs):
                adjacency[a, i + 4] = adjacency[i + 4, a] = 1
                adjacency[b, i + 4] = adjacency[i + 4, b] = 1
            self.register_buffer('pairs', pairs)
            self.register_buffer('laplacian', torch.diag(adjacency.sum(1)) - adjacency)
            self.projection = nn.Linear(embedding_dim * 100, hidden_dim)
        elif model_version == 'cnn':
            self.projection = nn.Conv1d(embedding_dim, hidden_dim, 4)
        else:
            self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.resolved_options = dict(model_version=model_version, embedding_dim=embedding_dim,
                                     hidden_dim=hidden_dim, dropout=dropout, pooling='masked_mean',
                                     normalization=normalization if model_version == 'tenet' else 'not_applicable')

    def forward(self, tokens):
        mask = tokens.ne(0)
        x = self.embedding(tokens) * mask.unsqueeze(-1)
        if self.version == 'tenet':
            # Each ordered window comprises four adjacent tokens, not a 2-D image.
            v = F.pad(x.transpose(1, 2), (0, 3)).unfold(2, 4, 1).permute(0, 2, 1, 3)
            valid = F.pad(mask, (0, 3)).unfold(1, 4, 1)
            edge_valid = valid[..., self.pairs[:, 0]] & valid[..., self.pairs[:, 1]]
            edge = (v[..., self.pairs[:, 0]] - v[..., self.pairs[:, 1]]).abs()
            edge = edge * edge_valid.unsqueeze(2)
            z = torch.cat((v, edge), dim=-1)
            graph = z.unsqueeze(-1) * self.laplacian * z.unsqueeze(-2)
            # Exclude padding nodes/edges from normalization, including window tails.
            node_valid = torch.cat((valid, edge_valid), dim=-1)
            active = (node_valid.unsqueeze(-1) & node_valid.unsqueeze(-2)).unsqueeze(2)
            if self.normalization != 'raw':
                count = active.sum((-1, -2), keepdim=True).clamp_min(1)
                center = (graph * active).sum((-1, -2), keepdim=True) / count
                graph = (graph - center) * active
                if self.normalization == 'standardized':
                    std = (graph.square().sum((-1, -2), keepdim=True) / count).clamp_min(1e-12).sqrt()
                    graph = graph / std
            features = F.relu(self.projection(graph.flatten(2)))
        elif self.version == 'cnn':
            features = F.relu(self.projection(F.pad(x.transpose(1, 2), (0, 3))).transpose(1, 2))
        else:
            pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            return self.classifier(self.dropout(F.relu(self.projection(pooled))))
        pooled = (features * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.classifier(self.dropout(pooled))

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embedding.weight.numel()
        return dict(total=total, embedding=embedding, backbone_and_head=total - embedding)


def matched_width(version, embedding_dim, num_classes, target):
    # Includes projection bias and classification head; embedding is shared budget.
    fan_in = embedding_dim * {'tenet': 100, 'cnn': 4, 'mean': 1}[version]
    return max(1, round((target - num_classes) / (fan_in + 1 + num_classes)))
