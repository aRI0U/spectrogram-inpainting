import torch.nn as nn


class MNISTEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        r"""

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
        """
        super(MNISTEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 2*in_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(2*in_channels, out_channels, kernel_size=5)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        r"""Forward pass of the encoder

        Parameters
        ----------
        inputs (torch.FloatTensor): batch of images, shape (batch_size, in_channels, height, width)

        Returns
        -------
        torch.FloatTensor: encoded images, shape (batch_size, out_channels, new_height, new_width)
        """
        x = self.conv1(inputs)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        return x
