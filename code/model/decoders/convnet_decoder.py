import torch.nn as nn


class ConvNetDecoder(nn.Module):
    """
        Creates a parametric DeconvNet model for the decoder.

        The model has the following architecture:

        One dense layer per element of parameter dense_layers:
            - # of neurons given by the item in dense_layers
            - each layer is followedby activation dense_activation

        One convolution transpose layer per element of parameter conv_channels:
            - # of channels given by the item in conv_channels
            - each convolution is followed by activation conv_activation
            - UpsamplingNearest2d between each pair of convolutions
    """

    def __init__(
            self,

            # the dimension of output pictures and
            # of the expected input latent variable
            # (default values correspond to MNIST)
            in_channels=50,
            out_height=28,
            out_width=28,
            out_channels=1,

            # architecture of the conv and dense layers
            dense_layers=None,
            conv_channels=None,
            conv_kernel=3,
            upsample_factor=2,
            batchnorm=True,

            # internal activation layers
            dense_activation='ReLU',
            conv_activation='ReLU'
    ):

        # nn.Module instantiate
        super().__init__()

        # store function parameters
        kwargs = {key: value for key, value in locals().items() if key not in ['self', '__class__']}
        vars(self).update(kwargs)

        # find the dimensions of the first convolution
        conv_dimensions = [(out_height, out_width)]
        height, width = out_height, out_width
        for i in range(len(conv_channels or []) - 1):
            height = height // upsample_factor
            width = width // upsample_factor
            conv_dimensions.append((height, width))
        conv_dimensions.reverse()
        self.in_height, self.in_width = conv_dimensions[0]

        # list convolution layers
        conv_sequence = []
        channels = (conv_channels or []) + [out_channels]
        for i in range(1, len(channels)):

            # conv => activation => batchnorm => upsample 
            # except after 
            conv_sequence.append(eval(f'nn.{conv_activation}()'))
            if batchnorm:
                conv_sequence.append(nn.BatchNorm2d(channels[i-1]))
            if i > 1:
                conv_sequence.append(nn.UpsamplingNearest2d(size=conv_dimensions[i-1]))
            conv_sequence.append(nn.ConvTranspose2d(
                in_channels=channels[i - 1],
                out_channels=channels[i],
                kernel_size=conv_kernel,
                padding=int(conv_kernel // 2)  # keeps the size as 'same'
            ))

        # list linear layers
        dense_sequence = []
        widths = [in_channels] + (dense_layers or [])
        for i in range(1, len(widths)):
            # add a dense layer and activation
            dense_sequence.append(nn.Linear(widths[i - 1], widths[i]))
            dense_sequence.append(eval(f'nn.{dense_activation}()'))
        # create the last layer
        dense_sequence.append(nn.Linear(widths[-1], channels[0]))
        
        # put layers in sequential objects for use in self.forward()
        self.conv = nn.Sequential(*conv_sequence)
        self.dense = nn.Sequential(*dense_sequence)

    
    @classmethod
    def mirror(cls, encoder):
        """ 
            Construct the mirror of the ConvNetEncoder 
        """
        return cls(
            # the dimension of output and input
            in_channels=encoder.out_channels,
            out_height=encoder.in_height,
            out_width=encoder.in_width,
            out_channels=encoder.in_channels,

            # architecture of the conv and dense layers
            dense_layers=None if not encoder.dense_layers else encoder.dense_layers[::-1],
            conv_channels=None if not encoder.conv_channels else encoder.conv_channels[::-1],
            conv_kernel=encoder.conv_kernel,
            upsample_factor=encoder.maxpool_kernel,
            batchnorm=encoder.batchnorm,

            # internal activation layers
            dense_activation=encoder.dense_activation,
            conv_activation=encoder.conv_activation
        )
    
    
    def forward(self, inputs):
        r"""
            Parameters
            ----------
            inputs (torch.Tensor): latent image, shape (batch_size, in_height, in_width, in_channels)

            Returns
            -------
            torch.Tensor: reconstructed spectrogram, shape (batch_size, out_channels, out_height, out_width)
        """

        # format to (N,H,W,C) for dense layers
        x = inputs.view(-1, self.in_height, self.in_width, self.in_channels)
        x = self.dense(x)
        
        # format to (N,C,H,W) format for convolutions
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)

        return x

