import torch.nn as nn


class ConvExpert(nn.Module):
    """Lightweight expert: three 3x3 conv layers, each followed by BatchNorm and ReLU.

    The final ReLU keeps outputs non-negative, which the scene-complexity entropy relies on.
    """

    def __init__(self, in_channels, out_channels=None, hidden_channels=None, num_layers=3):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        layers = []
        ch = in_channels
        for i in range(num_layers):
            nxt = out_channels if i == num_layers - 1 else hidden_channels
            layers += [
                nn.Conv2d(ch, nxt, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(nxt),
                nn.ReLU(inplace=True),
            ]
            ch = nxt
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)
