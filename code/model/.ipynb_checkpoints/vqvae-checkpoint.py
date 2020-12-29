import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import pytorch_lightning as pl

import model.encoders as encoders
import model.decoders as decoders
from model.quantizer import VectorQuantizer


class VQVAE(pl.LightningModule):
    def __init__(self,
                 architecture,
                 x_dim,
                 z_dim,
                 num_codewords,
                 commitment_cost,
                 lr=1e-3):
        r"""

        Parameters
        ----------
        architecture (str): architecture for the encoder/decoder
        x_dim (int): dimension of the input/output space
        z_dim (int): dimension of the latent space
        num_codewords (int): number of codewords
        commitment_cost (float): scaling for commitment loss
        lr (float):
        """
        super(VQVAE, self).__init__()

        self.save_hyperparameters()

        if architecture == "mnist":
            self.encoder = encoders.MNISTEncoder(x_dim, z_dim)
            self.decoder = decoders.MNISTDecoder(z_dim, x_dim)
        else:
            raise NotImplementedError("The following architecture is not implemented yet: " + architecture)

        self.quantizer = VectorQuantizer(num_codewords, z_dim, commitment_cost)

        # for graph logging
        self.example_input_array = torch.randn(1, x_dim, 32, 32)

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
        self.log('Reconstruction loss/Training', rec_loss)
        self.log('Quantization loss/Training', q_loss)
        self.log('Loss/Training', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, codes, q_loss = self(x)

        # compute loss
        rec_loss = F.mse_loss(x_hat, x)
        loss = rec_loss + q_loss

        # logging
        self.log('Reconstruction loss/Validation', rec_loss)
        self.log('Quantization loss/Validation', q_loss)
        self.log('Loss/Validation', loss)

        return loss

    def on_fit_start(self):
        metric_placeholder = {'val_loss': float('inf')}
        self.logger.log_hyperparams(self.hparams, metrics=metric_placeholder)

    def training_epoch_end(self, outputs):
        x, _ = next(iter(self.train_dataloader()))
        x_hat, _, _ = self(x)

        # display images in tensorboard
        self.logger.experiment.add_image("Originals/Training", make_grid(x.cpu().data), self.current_epoch)
        self.logger.experiment.add_image("Reconstructions/Training", make_grid(x_hat.cpu().data), self.current_epoch)

    def validation_epoch_end(self, outputs):
        x, _ = next(iter(self.val_dataloader()))
        x_hat, _, _ = self(x)

        # display images in tensorboard
        self.logger.experiment.add_image("Originals/Validation", make_grid(x.cpu().data), self.current_epoch)
        self.logger.experiment.add_image("Reconstructions/Validation", make_grid(x_hat.cpu().data), self.current_epoch)

        # log hyperparameters
        metrics_log = {'val_loss': torch.stack(outputs).mean()}
        self.logger.log_hyperparams(self.hparams, metrics=metrics_log)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
