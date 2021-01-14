"""Base class for VQ VAE Lightning module"""

import abc
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import pytorch_lightning as pl


class BaseVQVAE(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(self,
                 z_dim,
                 num_codewords,
                 commitment_cost,
                 lr=1e-3):
        r"""

        Parameters
        ----------
        z_dim (int): dimension of the latent space
        num_codewords (int): number of codewords
        commitment_cost (float): scaling for commitment loss
        lr (float): learning rate of the optimizer
        """
        super(BaseVQVAE, self).__init__()
        self.save_hyperparameters()

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

    def plot_codebook_usage(self, codes, training=False):
        r"""Computes the histogram of codebook usage and displays it to TensorBoard"""
        step = "Training" if training else "Validation"

        figure = plt.figure()
        plt.hist(codes.cpu().numpy(), bins=self.hparams.num_codewords - 1)

        self.logger.experiment.add_figure(
            f"Codebook usage/{step}",
            figure,
            self.current_epoch
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
