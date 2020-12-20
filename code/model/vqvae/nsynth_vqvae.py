import torch

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
        else:
            raise NotImplementedError(f"This architecture is not implemented yet: {architecture}")

        self.quantizer = VectorQuantizer(num_codewords, z_dim, commitment_cost)

        self.example_input_array = torch.randn(1, num_frequency_bins, num_timesteps)

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

    def training_epoch_end(self, outputs):
        x, _ = next(iter(self.train_dataloader()))
        x_hat, codes, _ = self(x)

        # display images in tensorboard
        self.logger.experiment.add_image("Originals/Training", x[0].cpu().data, self.current_epoch)
        self.logger.experiment.add_image("Reconstructions/Training", x_hat[0].cpu().data, self.current_epoch)
        self.plot_codebook_usage(codes, training=True)

    def validation_epoch_end(self, outputs):
        x, _ = next(iter(self.val_dataloader()))
        x_hat, codes, _ = self(x)

        # display images in tensorboard
        self.logger.experiment.add_image("Originals/Validation", x[0].cpu().data, self.current_epoch)
        self.logger.experiment.add_image("Reconstructions/Validation", x_hat[0].cpu().data, self.current_epoch)
        self.plot_codebook_usage(codes, training=False)

        # log hyperparameters
        metrics_log = {'val_loss': torch.stack(outputs).mean()}
        self.logger.log_hyperparams(self.hparams, metrics=metrics_log)