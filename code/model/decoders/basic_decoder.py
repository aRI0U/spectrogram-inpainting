import torch.nn as nn


class BasicDecoder(nn.Module):
    def __init__(self, num_frequency_bins, num_timesteps, z_dim):
        r"""# TODO: some parameters may be wrong or missins

        Parameters
        ----------
        num_frequency_bins (int)
        num_timesteps (int)
        z_dim (int)
        """
        super(BasicDecoder, self).__init__()

    def forward(self, inputs):
        r"""Forward pass of the decoder

        Parameters
        ----------
        inputs (torch.FloatTensor): discretized embeddings, shape (batch_size, ???)

        Returns
        -------
        torch.FloatTensor: reconstructed spectrograms, shape (batch_size, num_frequency_bins, num_timesteps)
        """
        pass
