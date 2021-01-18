import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_codewords, codewords_dim, commitment_cost, codebook_restart=False):
        r"""

        Parameters
        ----------
        num_codewords (int)
        codewords_dim (int)
        commitment_cost
        codebook_restart (bool): whether to add random restarts or not when codes are not used enough
        """
        super(VectorQuantizer, self).__init__()

        self.num_codewords = num_codewords
        self.codewords_dim = codewords_dim

        self.codebook_restart = codebook_restart
        self.sum_codebook_usage = torch.ones(self.num_codewords) / self.num_codewords
        self.number_samples = torch.ones(self.num_codewords, dtype=torch.int64)

        self.codewords = nn.Parameter(
            torch.randn(self.num_codewords, self.codewords_dim),
            requires_grad=True
        )

        self.commitment_cost = commitment_cost

    def forward(self, inputs, training=False):
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

        if training and self.codebook_restart:
            self.restart_unused_codes(flat_inputs)

        # compute pairwise distances (https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/6)
        distances = torch.pow(
            flat_inputs.unsqueeze(1) - self.codewords.unsqueeze(0),
            2
        ).sum(2)

        encoding_indices = torch.argmin(distances, dim=1)

        self.update_codebook_usage(encoding_indices)

        # quantized[i,j] = self.codewords[encoding_indices[i,j], j]
        quantized = torch.gather(
            self.codewords,
            0,
            encoding_indices.unsqueeze(1).expand(-1, self.codewords_dim)
        ).view(inputs_shape)

        # magic trick to copy gradients from inputs
        quantized = inputs + (quantized - inputs).detach()

        # quantization loss
        quantizing_loss = F.mse_loss(quantized.detach(), inputs)
        commitment_loss = F.mse_loss(quantized, inputs.detach())
        loss = quantizing_loss + self.commitment_cost * commitment_loss

        encoding_indices = encoding_indices.view(inputs_shape[:-1])

        return quantized, encoding_indices, loss

    @property
    def average_codebook_usage(self):
        return self.sum_codebook_usage / self.number_samples

    def update_codebook_usage(self, encoding_indices):
        r"""Randomly restart unused codes to avoid codebook collapse

        Parameters
        ----------
        encoding_indices (torch.LongTensor):
        """
        self.number_samples += 1
        bins, counts = torch.unique(encoding_indices, return_counts=True)
        usage = torch.zeros(self.num_codewords)
        usage[bins] = counts / len(encoding_indices)

        self.sum_codebook_usage += usage

    def restart_unused_codes(self, encoder_outputs):
        average_codebook_usage = self.sum_codebook_usage / self.number_samples
        to_restart = average_codebook_usage < 0.1 / self.num_codewords

        self.sum_codebook_usage[to_restart] = 1 / self.num_codewords
        self.number_samples[to_restart] = 1

        if to_restart.sum() > 0:
            print("restarting codes", *torch.arange(self.num_codewords)[to_restart].cpu().numpy())

            indices_new_codes = torch.randperm(len(encoder_outputs))[:to_restart.sum().item()]

            # self.codewords[to_restart][i] = encoder_outputs[indices_new_codes][i]
            self.codewords.data[to_restart] = encoder_outputs[indices_new_codes]

            if self.codewords.grad is not None:
                self.codewords.grad[to_restart] = 0
