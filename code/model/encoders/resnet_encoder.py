import torch.nn as nn


class ConvNetEncoder2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNetEncoder2, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x).permute(0, 2, 3, 1).contiguous()  # channels last
