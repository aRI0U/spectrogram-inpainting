import torch.nn as nn


class MNISTDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        r"""

        Parameters
        ----------
        in_channels
        out_channels
        """
        super(MNISTDecoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=5)
        self.conv2 = nn.ConvTranspose2d(in_channels//2, out_channels, kernel_size=5)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        r"""Forward pass of the decoder

        Parameters
        ----------
        inputs

        Returns
        -------

        """
        x = inputs.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x
