import torch.nn as nn

class DeconvModel(nn.Module):
    
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

        # the dimension of the input pictures and
        # of the expected output latent variable
        # (default values correspond to MNIST)
        input_dim=10,        
        output_height=28, 
        output_width=28,
        output_channels=1,

        # architecture of the conv and dense layers
        dense_layers=None,
        conv_channels=None, 
        conv_kernel=3,
        upsample_factor=2,

        # internal activation layers
        dense_activation='ReLU',
        conv_activation='ReLU'
    ):
        
        # nn.Module instantiate
        super().__init__()

        # store useful parameters
        self._input = input_dim

        # find the dimensions of the first convolution
        conv_dimensions = [(output_height, output_width)]
        height, width = output_height, output_width
        for i in range(len(conv_channels or []) - 1):
            height = height // upsample_factor ** 2
            width = width // upsample_factor ** 2
            conv_dimensions.append((height, width))
        conv_dimensions.reverse()
        self._conv_in = (output_channels if conv_channels is None else conv_channels[0], *conv_dimensions[0])


        # list linear layers
        dense_sequence = []
        widths = [input_dim] + (dense_layers or [])
        for i in range(1, len(widths)):

            # add a dense layer and activation
            dense_sequence.append(nn.Linear(widths[i-1], widths[i]))
            dense_sequence.append(eval(f'nn.{dense_activation}()'))

        # create the last layer
        dense_sequence.append(nn.Linear(widths[-1], self._conv_in[0] * self._conv_in[1] * self._conv_in[2]))


        # list convolution layers
        conv_sequence = []
        if conv_channels:
            channels = conv_channels + [output_channels]
            for i in range(1, len(channels)):
                
                # add a convolution layer
                conv_sequence.append(nn.ConvTranspose2d(
                    in_channels=channels[i-1],
                    out_channels=channels[i],
                    kernel_size=conv_kernel,
                    padding=int(conv_kernel // 2)  # keeps the size as 'same'
                ))
                
                # add an activation  
                conv_sequence.append(eval(f'nn.{conv_activation}()'))

                # add upsampling
                if i < len(channels) - 1:
                    conv_sequence.append(nn.UpsamplingNearest2d(size=conv_dimensions[i]))


        # put layers in sequential objects for use in self.forward()
        self.conv = nn.Sequential(*conv_sequence)
        self.dense = nn.Sequential(*dense_sequence)
        
        
    def forward(self, inputs):
                    
        # format for dense layers
        x = inputs.view(-1, self._input)
        
        # apply dense layers if any
        x = self.dense(x)
        
        # format for convolutions
        x = x.view(-1, *self._conv_in)
        
        # apply convolutions if any
        outputs = self.conv(x)
        
        return outputs