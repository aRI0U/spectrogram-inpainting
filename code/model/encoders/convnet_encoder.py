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
        if not conv_channels:
            self._conv_out = input_height * input_width * input_channels
        else:
            channels = [input_channels] + conv_channels
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
                else:
                    conv_sequence.append(nn.Flatten())

            # determine the number of outputs from the convolution layers
            self._conv_out = input_height * input_width * channels[-1]
            for i in range(len(conv_channels) - 1):
                self._conv_out = self._conv_out // maxpool_kernel ** 2

        # linear layers
        dense_sequence = []
        if dense_layers is None:
            # default last layer
            self.l_final = nn.Linear(self._conv_out, output_dim)
        else:
            widths = [self._conv_out] + dense_layers
            for i in range(1, len(widths)):
                # add a dense layer and activation
                dense_sequence.append(nn.Linear(widths[i - 1], widths[i]))
                dense_sequence.append(eval(f'nn.{dense_activation}()'))

            # create the last layer
            self.l_final = nn.Linear(widths[-1], output_dim)

        # put layers in sequential objects for use in self.forward()
        self.conv = nn.Sequential(*conv_sequence)
        self.dense = nn.Sequential(*dense_sequence)

    def forward(self, inputs):
        # format for convolution
        x = inputs.view(-1, self._channels, self._height, self._width)

        # apply convolutions if any
        if self.conv:
            x = self.conv(x)

        # format for dense layers
        x = x.view(-1, self._conv_out)

        # apply dense layers if any
        if self.dense:
            x = self.dense(x)

        # final layer
        outputs = self.l_final(x)

        return outputs
