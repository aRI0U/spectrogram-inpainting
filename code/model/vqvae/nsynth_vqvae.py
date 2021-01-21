import torch
import torch.nn.functional as F
import torchaudio.transforms

import model.encoders as encoders
import model.decoders as decoders
from model.quantizers import VectorQuantizer
from model.vqvae.base_vqvae import BaseVQVAE
from utils.co2_tracker import CO2Tracker


class NSynthVQVAE(BaseVQVAE):
    def __init__(self,
                 architecture,
                 nfft,
                 win_length,
                 z_dim,
                 num_codewords,
                 commitment_cost,
                 codebook_restart,
                 **optimizer_kwargs):
        r"""

        Parameters
        ----------
        architecture (str)
        nfft (int)
        win_length (int)
        z_dim (int)
        num_codewords (int)
        commitment_cost (float)
        optimizer_kwargs (dict)
        """
        super(NSynthVQVAE, self).__init__(z_dim,
                                          num_codewords,
                                          commitment_cost,
                                          **optimizer_kwargs)

        num_frequency_bins = nfft // 2 + 1
        num_timesteps = 64000 * 2 // win_length + 1

        if architecture == 'basic':
            self.encoder = encoders.BasicEncoder(num_frequency_bins, num_timesteps, z_dim)
            self.decoder = decoders.BasicDecoder(num_frequency_bins, num_timesteps, z_dim)
        elif architecture == 'convnet':
            self.encoder = encoders.ConvNetEncoder(
                in_height=num_frequency_bins,
                in_width=num_timesteps,
                in_channels=1,
                out_channels=z_dim,
                conv_channels=[4, 8, 16],
                dense_layers=[16]
            )
            self.decoder = decoders.ConvNetDecoder.mirror(self.encoder)
        else:
            raise NotImplementedError(f"This architecture is not implemented yet: {architecture}")

        self.quantizer = VectorQuantizer(num_codewords, z_dim, commitment_cost, codebook_restart)

        self.example_input_array = torch.randn(1, 1, num_frequency_bins, num_timesteps)

        self.inverse_transform = torchaudio.transforms.GriffinLim(n_fft=nfft, win_length=win_length, n_iter=512)

        self.tracker = CO2Tracker()

        print(self)

    def forward(self, x, training=True):
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
        z_q, codes, q_loss = self.quantizer(z_e, training=training)

        # 3. decode
        x_hat = self.decoder(z_q)

        return x_hat, codes, q_loss

    # %% engineering
    def on_train_start(self):
        self.tracker.start()

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        x_hat, codes, q_loss = self(x, training=True)

        # compute loss
        rec_loss = F.mse_loss(x_hat, x)
        loss = rec_loss + q_loss

        # logging
        self.log('Reconstruction loss/Training', rec_loss)
        self.log('Quantization loss/Training', q_loss)
        self.log('Loss/Training', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.to(self.device)
        x_hat, codes, q_loss = self(x, training=False)

        # compute loss
        rec_loss = F.mse_loss(x_hat, x)
        loss = rec_loss + q_loss

        # logging
        self.log('Reconstruction loss/Validation', rec_loss)
        self.log('Quantization loss/Validation', q_loss)
        self.log('Loss/Validation', loss)

        return loss

    def training_epoch_end(self, outputs):
        self.tracker.record()

        x = next(iter(self.train_dataloader()))[:8].to(self.device)
        x_hat, codes, _ = self(x)

        # display images in tensorboard
        self.logger.experiment.add_images("Originals/Training", x.cpu().data, self.current_epoch)
        self.logger.experiment.add_images("Reconstructions/Training", x_hat.cpu().data, self.current_epoch)

        # display codebook usage
        self.plot_codebook_usage(codes, training=True)

        # send audio
        idx = torch.randint(0, 8, (1,))
        self.logger.experiment.add_audio(
            "Originals/Training",
            self.inverse_transform(x[idx]).cpu().data,
            self.current_epoch
        )
        self.logger.experiment.add_audio(
            "Reconstructions/Training",
            self.inverse_transform(x_hat[idx]).cpu().data,
            self.current_epoch
        )
        
        # save audio
        torchaudio.save("output/training_{}.wav".format(idx),self.inverse_transform(x_hat[idx]),16e3)

        # display energy consumption
        self.logger.experiment.add_scalars(
            "Consumption/Power",
            self.tracker.power_stats(),
            self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "Consumption/CO2 equivalent",
            self.tracker.co2_equivalent,
            self.current_epoch
        )

    def validation_epoch_end(self, outputs):
        x = next(iter(self.val_dataloader()))[:8].to(self.device)
        x_hat, codes, _ = self(x)

        # display images in tensorboard
        self.logger.experiment.add_images("Originals/Validation", x.cpu().data, self.current_epoch)
        self.logger.experiment.add_images("Reconstructions/Validation", x_hat.cpu().data, self.current_epoch)

        # display codebook usage
        self.plot_codebook_usage(codes, training=False)

        # send audio
        idx = torch.randint(0, 8, (1,))
        self.logger.experiment.add_audio(
            "Originals/Validation",
            self.inverse_transform(x[idx]).cpu().data,
            self.current_epoch
        )
        self.logger.experiment.add_audio(
            "Reconstructions/Validation",
            self.inverse_transform(x_hat[idx]).cpu().data,
            self.current_epoch
        )

        # save audio
        torchaudio.save("output/validation_{}.wav".format(idx),self.inverse_transform(x_hat[idx]),16e3)

        # log hyperparameters
        metrics_log = {'val_loss': torch.stack(outputs).mean()}
        self.logger.log_hyperparams(self.hparams, metrics=metrics_log)
