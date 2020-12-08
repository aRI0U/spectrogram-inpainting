import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import pytorch_lightning as pl

from model.encoder import Encoder
from model.decoder import Decoder
from model.quantizer import VectorQuantizer


class VQVAE(pl.LightningModule):
    def __init__(self, x_dim, z_dim, num_codewords, commitment_cost):
        r"""

        Parameters
        ----------
        x_dim (int): dimension of the input/output space
        z_dim (int): dimension of the latent space
        num_codewords (int): number of codewords
        commitment_cost (float): scaling for commitment loss
        """
        super(VQVAE, self).__init__()

        self.encoder = Encoder(x_dim, z_dim)
        self.decoder = Decoder(z_dim, x_dim)

        self.quantizer = VectorQuantizer(num_codewords, z_dim, commitment_cost)

    def forward(self, x):
        r"""Forward pass of VQ-VAE

        Parameters
        ----------
        x (torch.FloatTensor): input, shape (B, C, H, W)

        Returns
        -------
        x_hat (torch.FloatTensor): reconstructed input, shape (B, C, H, W)
        codes (torch.LongTensor): corresponding codes, shape (B)
        q_loss (torch.FloatTensor): quantization loss, shape (1)
        """
        # 1. encode
        z_e = self.encoder(x)

        # 2. quantize
        z_q, codes, q_loss = self.quantizer(z_e)

        # 3. decode
        x_hat = self.decoder(z_q)

        return x_hat, codes, q_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, codes, q_loss = self(x)

        # compute loss
        rec_loss = F.mse_loss(x_hat, x)
        loss = rec_loss + q_loss

        # logging
        self.log('Reconstruction loss [train]', rec_loss)
        self.log('Quantization loss [train]', q_loss)
        self.log('Loss [train]', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, codes, q_loss = self(x)

        # compute loss
        rec_loss = F.mse_loss(x_hat, x)
        loss = rec_loss + q_loss

        # logging
        self.log('Reconstruction loss [val]', rec_loss)
        self.log('Quantization loss [val]', q_loss)
        self.log('Loss [val]', loss)

        return loss

    def validation_epoch_end(self, outputs):
        x, _ = next(iter(self.val_dataloader()))
        x_hat, _, _ = self(x)

        # display images in tensorboard
        self.logger.experiment.add_image("Originals", make_grid(x.cpu().data), self.current_epoch)
        self.logger.experiment.add_image("Reconstructions", make_grid(x_hat.cpu().data), self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
