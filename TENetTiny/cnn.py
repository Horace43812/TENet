"""Small conventional CNN baseline with configurable channel widths."""
from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels=1, image_size=28, num_classes=10,
                 channels=(24, 3), head='linear', head_hidden=48, classifier_dropout=0.):
        super().__init__()
        if not channels or min(channels) < 1 or min(in_channels, num_classes, head_hidden) < 1:
            raise ValueError('CNN dimensions must be positive')
        if image_size < 2 ** len(channels):
            raise ValueError('Image too small for the number of CNN pooling stages')
        if head not in ('linear', 'linear-relu', 'mlp') or not 0 <= classifier_dropout < 1:
            raise ValueError('Invalid CNN classifier options')
        blocks = []
        previous = in_channels
        for width in channels:
            blocks.extend([nn.Conv2d(previous, width, 3, padding=1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)])
            previous = width
        self.features = nn.Sequential(*blocks)
        count = previous * (image_size // 2 ** len(channels)) ** 2
        classifier = [nn.Flatten()]
        if head == 'mlp':
            classifier.extend([nn.Linear(count, head_hidden), nn.ReLU(),
                               nn.Dropout(classifier_dropout), nn.Linear(head_hidden, num_classes)])
        else:
            classifier.extend([nn.Dropout(classifier_dropout), nn.Linear(count, num_classes)])
            if head == 'linear-relu':
                classifier.append(nn.ReLU())
        self.classifier = nn.Sequential(*classifier)
        self.input_shape = (in_channels, image_size, image_size)
        self.resolved_options = dict(channels=list(channels), head=head, head_hidden=head_hidden,
                                     classifier_dropout=classifier_dropout, normalization='not_applicable',
                                     batch_normalization=True)

    def forward(self, x):
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError('Unexpected input shape')
        return self.classifier(self.features(x))
