import torch.nn as nn


class ConvNetEncoder(nn.Module):
    """
        Creates a parametric Convnet model for the encoder.

        The model has the following architecture:

        One convolution layer per element of parameter conv_channels:
            - # of channels given by the item in conv_channels
            - each convolution is followed by activation conv_activation
            - maxpool2D between each pair of convolutions

        One dense layer per element of parameter dense_layers:
            - # of neurons given by the item in dense_layers
            - each layer is followedby activation dense_activation
    """

    def __init__(
            self,

            # the dimension of the input pictures and
            # of the expected output latent variable
            # (default values correspond to MNIST)
            input_height=28,
            input_width=28,
            input_channels=1,
            output_channels=50,

            # architecture of the conv and dense layers
            conv_channels=None,
            dense_layers=None,
            conv_kernel=3,
            maxpool_kernel=2,
            batchnorm=True,

            # internal activation layers
            conv_activation='ReLU',
            dense_activation='ReLU'
    ):

        # nn.Module instantiate
        super().__init__()

        # store useful parameters
        self._height = input_height
        self._width = input_width
        self._channels = input_channels

        # list convolution layers
        conv_sequence = []
        channels = [input_channels] + (conv_channels or [])
        for i in range(1, len(channels)):

            # conv => maxpool => activation => batchnorm
            conv_sequence.append(nn.Conv2d(
                in_channels=channels[i - 1],
                out_channels=channels[i],
                kernel_size=conv_kernel,
                padding=int(conv_kernel // 2)  # keeps the size as 'same'
            ))
            if i < len(channels) - 1:
                conv_sequence.append(nn.MaxPool2d(kernel_size=maxpool_kernel))
            conv_sequence.append(eval(f'nn.{conv_activation}()'))
            if batchnorm:
                conv_sequence.append(nn.BatchNorm2d(channels[i]))

        # size of image after downscaling
        h, w = input_height, input_width
        for i in range(len(conv_channels or []) - 1):
            h, w = h // maxpool_kernel, w // maxpool_kernel
        self.h_out = h
        self.w_out = w

        # list linear layers
        dense_sequence = []
        widths = channels[-1:] + (dense_layers or [])
        for i in range(1, len(widths)):
            # add a dense layer and activation
            dense_sequence.append(nn.Linear(widths[i - 1], widths[i]))
            dense_sequence.append(eval(f'nn.{dense_activation}()'))
        # create the last layer
        dense_sequence.append(nn.Linear(widths[-1], output_channels))

        # put layers in sequential objects for use in self.forward()
        self.conv = nn.Sequential(*conv_sequence)
        self.dense = nn.Sequential(*dense_sequence)
        self.flatten = nn.Flatten(start_dim=0, end_dim=2)

    def forward(self, inputs):
        # format to (N,C,H,W) format for convolutions
        x = inputs.view(-1, self._channels, self._height, self._width)
        x = self.conv(x)
        
        # format to (N,H,W,C) format for dense layers
        x = x.permute(0, 2, 3, 1)
        x = self.dense(x)
        
        # format to (N*H*W,C) format for quantization
        outputs = self.flatten(x)
        
        return outputs