import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codewords, codewords_dim, commitment_cost, ema_decay, codebook_restart=False):
        r"""

        Parameters
        ----------
        num_codewords (int)
        codewords_dim (int)
        commitment_cost (float):
        ema_decay (float): decay for EMA smooth updates
        codebook_restart (bool): whether to add random restarts or not when codes are not used enough
        """
        super(VectorQuantizerEMA, self).__init__()

        self.num_codewords = num_codewords
        self.codewords_dim = codewords_dim
        self.register_buffer('codewords', torch.randn(self.num_codewords, self.codewords_dim))

        # codebook usage related variables
        self.codebook_restart = codebook_restart
        self.register_buffer('smoothed_codebook_usage', torch.ones(self.num_codewords) / self.num_codewords)

        # EMA related variables
        self.ema_decay = ema_decay

        self.commitment_cost = commitment_cost

        self.initialized = False

    def forward(self, inputs):
        r"""

        Parameters
        ----------
        inputs (torch.FloatTensor): shape (*, codewords_dim)

        Returns
        -------
        torch.FloatTensor: quantized inputs, shape (*, codewords_dim)
        torch.LongTensor: indices of quantized inputs in the codebook, shape (*)
        torch.FloatTensor: quantization loss
        """
        inputs_shape = inputs.size()
        flat_inputs = inputs.view(-1, self.codewords_dim)

        if self.training:
            if not self.initialized:
                self.initialize_codes(flat_inputs)
                self.initialized = True
            if self.codebook_restart:
                self.restart_unused_codes(flat_inputs)

        # compute pairwise distances (https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/6)
        distances = torch.pow(
            flat_inputs.unsqueeze(1) - self.codewords.unsqueeze(0),
            2
        ).sum(2)

        encoding_indices = torch.argmin(distances, dim=1)

        # quantized[i,j] = self.codewords[encoding_indices[i,j], j]
        quantized = torch.gather(
            self.codewords,
            0,
            encoding_indices.unsqueeze(1).expand(-1, self.codewords_dim)
        ).view(inputs_shape)

        if self.training:
            self.update_codebook_ema(encoding_indices, flat_inputs)

        # quantization loss
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * commitment_loss

        encoding_indices = encoding_indices.view(inputs_shape[:-1])

        return quantized, encoding_indices, loss

    def update_codebook_ema(self, encoding_indices, inputs):
        r"""Update codewords and smoothed codebook usage using Exponential Moving Averages (EMA)

        Parameters
        ----------
        encoding_indices (torch.LongTensor):
        inputs (torch.Tensor):
        """
        num_samples = len(inputs)
        smoothed_cluster_sizes = num_samples * self.smoothed_codebook_usage.unsqueeze(1)
        unnormalized_codewords = smoothed_cluster_sizes * self.codewords

        # sum_inputs[i] is the sum of all inputs whose closest codeword is self.codewords[i]
        sum_inputs = torch.zeros_like(self.codewords)
        # TODO: avoid this ugly `for` loop
        for i in range(self.num_codewords):
            sum_inputs[i] = inputs[encoding_indices == i].sum(dim=0)

        unnormalized_codewords = self.ema_decay * unnormalized_codewords + (1 - self.ema_decay) * sum_inputs

        # update smoothed codebook usage
        bins, counts = torch.unique(encoding_indices, return_counts=True)
        codebook_usage = torch.zeros(self.num_codewords, device=bins.device)
        codebook_usage[bins] = counts.to(torch.float32)

        smoothed_cluster_sizes = self.ema_decay * smoothed_cluster_sizes + \
                                 (1 - self.ema_decay) * codebook_usage.unsqueeze(1)
        self.smoothed_codebook_usage = smoothed_cluster_sizes.squeeze(1) / num_samples

        # finally update codebook
        self.codewords = unnormalized_codewords / smoothed_cluster_sizes

        # DEBUG:
        unused = codebook_usage == 0
        num_unused = unused.sum().item()
        if num_unused > 0:
            print("unused codes", *torch.arange(self.num_codewords)[unused].cpu().numpy())

    def initialize_codes(self, inputs):
        r"""Initialize codebook with outputs of the encoder

        Parameters
        ----------
        inputs (torch.Tensor):  flattened inputs, shape (N, codewords_dim)

        """
        random_indices = torch.randperm(len(inputs))[:self.num_codewords]
        self.codewords = inputs[random_indices]

    def restart_unused_codes(self, inputs):
        r"""Reinitialize unused codes with outputs of the encoder to avoid codebook collapse

        Parameters
        ----------
        inputs (torch.Tensor): flattened outputs of the encoder, shape (N, codewords_dim)
        """
        to_restart = self.smoothed_codebook_usage < 0.5 / self.num_codewords

        self.smoothed_codebook_usage[to_restart] = 1 / self.num_codewords

        num_to_restart = to_restart.sum().item()
        if num_to_restart > 0:
            print("restarting codes", *torch.arange(self.num_codewords)[to_restart].cpu().numpy())

            indices_new_codes = torch.randperm(len(inputs))[:num_to_restart]

            # self.codewords[to_restart][i] = encoder_outputs[indices_new_codes][i]
            self.codewords[to_restart] = inputs[indices_new_codes]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_codewords = 59
    codewords_dim = 2

    num_epochs = 500

    quantizer = VectorQuantizerEMA(num_codewords, codewords_dim, 0.15, 0.95, codebook_restart=True)

    with torch.no_grad():
        z_e = torch.randn(7, 3, 4, codewords_dim)
        flat_inputs = z_e.view(-1, codewords_dim).cpu().numpy()

        codewords_list = []

        for _ in range(num_epochs):
            z_q, codes, loss = quantizer(z_e)
            codewords_list.append(quantizer.codewords.cpu().numpy())

    for i, codewords in enumerate(codewords_list):
        alpha = (i+1) / num_epochs
        plt.scatter(*codewords.T, c='r', alpha=alpha)

    plt.scatter(*flat_inputs.T, c='b', alpha=0.5)

    plt.show()
