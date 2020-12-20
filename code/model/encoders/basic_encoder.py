import torch.nn as nn


class BasicEncoder(nn.Module):
    def __init__(self, num_frequency_bins, num_timesteps, z_dim):
        r"""# TODO: some parameters may be wrong or missins

        Parameters
        ----------
        num_frequency_bins (int)
        num_timesteps (int)
        z_dim (int)
        """
        super(BasicEncoder, self).__init__()

    def forward(self, inputs):
        r"""Forward pass of the encoder

        Parameters
        ----------
        inputs (torch.FloatTensor): batch of spectrograms, shape (batch_size, num_frequency_bins, num_timestesp)

        Returns
        -------
        torch.FloatTensor: ..., shape (B, ???)  # TODO: what is the output of this
        """
        pass
