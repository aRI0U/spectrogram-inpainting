import torch
import torch.nn.functional as F

import model.encoders as encoders
import model.decoders as decoders
from model.quantizers import VectorQuantizer
from model.vqvae.base_vqvae import BaseVQVAE


class NSynthVQVAE(BaseVQVAE):
    def __init__(self,
                 architecture,
                 num_frequency_bins,
                 num_timesteps,
                 z_dim,
                 num_codewords,
                 commitment_cost,
                 **optimizer_kwargs):
        r"""

        Parameters
        ----------
        architecture (str)
        num_frequency_bins (int)
        num_timesteps (int)
        z_dim (int)
        num_codewords (int)
        commitment_cost (float)
        optimizer_kwargs (dict)
        """
        super(NSynthVQVAE, self).__init__(z_dim,
                                          num_codewords,
                                          commitment_cost,
                                          **optimizer_kwargs)

        if architecture == 'basic':
            self.encoder = encoders.BasicEncoder(num_frequency_bins, num_timesteps, z_dim)
            self.decoder = decoders.BasicDecoder(num_frequency_bins, num_timesteps, z_dim)
        elif architecture == 'convnet':
            self.encoder = encoders.ConvNetEncoder(
                input_height=num_frequency_bins,
                input_width=num_timesteps,
                input_channels=1,
                output_dim=z_dim
            )
            self.decoder = decoders.ConvNetDecoder(
                input_dim=z_dim,
                output_height=num_frequency_bins,
                output_width=num_timesteps,
                output_channels=1
            )
        else:
            raise NotImplementedError(f"This architecture is not implemented yet: {architecture}")

        self.quantizer = VectorQuantizer(num_codewords, z_dim, commitment_cost)

        self.example_input_array = torch.randn(1, 1, num_frequency_bins, num_timesteps)

    def forward(self, x):
        r"""Forward pass of VQ-VAE

        Parameters
        ----------
        x (torch.FloatTensor): batch of spectrograms, shape (batch_size, num_frequency_bins, num_timesteps)

        Returns
        -------
        x_hat (torch.FloatTensor): reconstructed spectrograms, shape (batch_size, num_frequency_bins, num_timesteps)
        codes (torch.LongTensor): encoding indices, shape (???)
        q_loss (torch.FloatTensor): quantization loss, shape (1)
        """
        # TODO: improve this implementation
        # 1. encode
        z_e = self.encoder(x)

        # 2. quantize
        z_q, codes, q_loss = self.quantizer(z_e)

        # 3. decode
        x_hat = self.decoder(z_q)

        return x_hat, codes, q_loss

    def training_step(self, batch, batch_idx):
        x_hat, codes, q_loss = self(batch)

        # compute loss
        rec_loss = F.mse_loss(x_hat, batch)
        loss = rec_loss + q_loss

        # logging
        self.log('Reconstruction loss/Training', rec_loss)
        self.log('Quantization loss/Training', q_loss)
        self.log('Loss/Training', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x_hat, codes, q_loss = self(batch)

        # compute loss
        rec_loss = F.mse_loss(x_hat, batch)
        loss = rec_loss + q_loss

        # logging
        self.log('Reconstruction loss/Validation', rec_loss)
        self.log('Quantization loss/Validation', q_loss)
        self.log('Loss/Validation', loss)

        return loss

    def training_epoch_end(self, outputs):
        x = next(iter(self.train_dataloader()))[:8]
        x_hat, codes, _ = self(x)

        # display images in tensorboard
        self.logger.experiment.add_images("Originals/Training", x.cpu().data, self.current_epoch)
        self.logger.experiment.add_images("Reconstructions/Training", x_hat.cpu().data, self.current_epoch)

        # display codebook usage
        self.plot_codebook_usage(codes, training=True)

        # send audio


    def validation_epoch_end(self, outputs):
        x = next(iter(self.val_dataloader()))[:8]
        x_hat, codes, _ = self(x)

        # display images in tensorboard
        self.logger.experiment.add_images("Originals/Validation", x.cpu().data, self.current_epoch)
        self.logger.experiment.add_images("Reconstructions/Validation", x_hat.cpu().data, self.current_epoch)

        # display codebook usage
        self.plot_codebook_usage(codes, training=False)

        # send audio

        # log hyperparameters
        metrics_log = {'val_loss': torch.stack(outputs).mean()}
        self.logger.log_hyperparams(self.hparams, metrics=metrics_log)