import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_codewords, codewords_dim, commitment_cost):
        r"""

        Parameters
        ----------
        num_codewords (int)
        codewords_dim (int)
        commitment_cost
        """
        super(VectorQuantizer, self).__init__()

        self.num_codewords = num_codewords
        self.codewords_dim = codewords_dim

        self.codewords = nn.Parameter(
            torch.rand(self.num_codewords, self.codewords_dim),
            requires_grad=True
        )

        self.commitment_cost = commitment_cost

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
        # flatten inputs: B, C, H, W -> B*H*W, C
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs_shape = inputs.size()
        flat_inputs = inputs.view(-1, self.codewords_dim)

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
        encoding_indices = encoding_indices.view(inputs_shape[:-1])

        # quantization loss
        quantizing_loss = F.mse_loss(quantized.detach(), inputs)
        commitment_loss = F.mse_loss(quantized, inputs.detach())
        loss = quantizing_loss + self.commitment_cost * commitment_loss

        # magic trick to copy gradients from inputs
        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 3, 1, 2), encoding_indices, loss
