import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        r"""

        Parameters
        ----------
        in_channels
        out_channels
        """
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels//2, out_channels, kernel_size=3)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        r"""Forward pass of the encoder

        Parameters
        ----------
        inputs

        Returns
        -------

        """
        x = self.conv1(inputs)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        return x
