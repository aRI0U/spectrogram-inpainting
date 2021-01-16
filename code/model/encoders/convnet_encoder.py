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
            output_dim=10,

            # architecture of the conv and dense layers
            conv_channels=None,
            dense_layers=None,
            conv_kernel=3,
            maxpool_kernel=2,

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

            # add a convolution layer
            conv_sequence.append(nn.Conv2d(
                in_channels=channels[i - 1],
                out_channels=channels[i],
                kernel_size=conv_kernel,
                padding=int(conv_kernel // 2)  # keeps the size as 'same'
            ))

            # add an activation  
            conv_sequence.append(eval(f'nn.{conv_activation}()'))

            # add maxpool
            if i < len(channels) - 1:
                conv_sequence.append(nn.MaxPool2d(kernel_size=maxpool_kernel))
                
        # determine the number of outputs from the convolution layers
        self._conv_out = input_height * input_width * channels[-1]
        h, w = input_height, input_width
        for i in range(len(conv_channels or []) - 1):
            h, w = h // maxpool_kernel, w // maxpool_kernel
        self._conv_out = h * w * channels[-1]
    
        # linear layers
        dense_sequence = []
        widths = [self._conv_out] + (dense_layers or [])
        for i in range(1, len(widths)):
            # add a dense layer and activation
            dense_sequence.append(nn.Linear(widths[i - 1], widths[i]))
            dense_sequence.append(eval(f'nn.{dense_activation}()'))
        # create the last layer
        dense_sequence.append(nn.Linear(widths[-1], output_dim))

        # put layers in sequential objects for use in self.forward()
        self.conv = nn.Sequential(*conv_sequence)
        self.dense = nn.Sequential(*dense_sequence)

        
    def forward(self, inputs):
        # format for convolution
        x = inputs.view(-1, self._channels, self._height, self._width)

        # apply convolutions if any
        x = self.conv(x)
        
        # format for dense layers
        x = x.view(-1, self._conv_out)

        # apply dense layers if any
        outputs = self.dense(x)

        return outputs

