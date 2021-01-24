import torch
import torch.nn.functional as F
import torchaudio.transforms

import model.encoders as encoders
import model.decoders as decoders
import model.quantizers as quantizers
from model.vqvae.base_vqvae import BaseVQVAE


class NSynthVQVAE(BaseVQVAE):
    def __init__(self,
                 architecture,
                 nfft,
                 win_length,
                 z_dim,
                 num_codewords,
                 commitment_cost,
                 codebook_restart=False,
                 use_ema=False,
                 ema_decay=None,
                 gpus=[],
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
        codebook_restart (bool):
        use_ema (bool):
        ema_decay (Optional[float]):
        optimizer_kwargs (dict)
        """
        super(NSynthVQVAE, self).__init__(z_dim,
                                          num_codewords,
                                          commitment_cost,
                                          gpus,
                                          **optimizer_kwargs)

        num_frequency_bins = nfft // 2 + 1
        num_timesteps = 64000 * 2 // win_length + 1

        if architecture == 'basic':
            self.encoder = encoders.BasicEncoder(num_frequency_bins, num_timesteps, z_dim)
            self.decoder = decoders.BasicDecoder(num_frequency_bins, num_timesteps, z_dim)
        elif architecture == 'mnist_like':
            self.encoder = encoders.MNISTEncoder(1, z_dim)
            self.decoder = decoders.MNISTDecoder(z_dim, 1)
        elif architecture == 'convnet':
            self.encoder = encoders.ConvNetEncoder(
                in_height=num_frequency_bins,
                in_width=num_timesteps,
                in_channels=1,
                out_channels=z_dim,
                conv_channels=[8, 16, 32, 64],
                dense_layers=[64]
            )
            self.decoder = decoders.ConvNetDecoder.mirror(self.encoder)
        elif architecture == 'convnet2':
            self.encoder = encoders.ConvNetEncoder2(
                in_channels=1,
                out_channels=z_dim
            )
            self.decoder = decoders.ConvNetDecoder2(
                in_channels=z_dim,
                out_channels=1,
                in_height=num_frequency_bins,
                in_width=num_timesteps
            )
        else:
            raise NotImplementedError(f"This architecture is not implemented yet: {architecture}")

        if use_ema:
            self.quantizer = quantizers.VectorQuantizerEMA(
                num_codewords=num_codewords,
                codewords_dim=z_dim,
                commitment_cost=commitment_cost,
                ema_decay=ema_decay,
                codebook_restart=codebook_restart
            )
        else:
            self.quantizer = quantizers.VectorQuantizer(num_codewords, z_dim, commitment_cost, codebook_restart)

        self.example_input_array = torch.randn(1, 1, num_frequency_bins, num_timesteps)

        self.inverse_transform = torchaudio.transforms.GriffinLim(n_fft=nfft, win_length=win_length, n_iter=512)

    # %% engineering
    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        x_hat, codes, q_loss = self(x)

        # compute loss
        # WARNING: temporary stuff
        rec_loss = F.mse_loss(x_hat, x)
        loss = rec_loss + q_loss

        # logging
        self.log('Reconstruction loss/Training', rec_loss)
        self.log('Quantization loss/Training', q_loss)
        self.log('Loss/Training', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.to(self.device)
        x_hat, codes, q_loss = self(x)

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
        idx = torch.randint(0, 8, ())
        self.log_audio(x[idx], x_hat[idx], "Training")

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
        idx = torch.randint(0, 8, ())
        self.log_audio(x[idx], x_hat[idx], "Validation")

        # log hyperparameters
        metrics_log = {'val_loss': torch.stack(outputs).mean()}
        self.logger.log_hyperparams(self.hparams, metrics=metrics_log)

    def log_audio(self, original, reconstruction, step, rate=16000):
        original_audio = self.inverse_transform(original).cpu()
        reconstructed_audio = self.inverse_transform(reconstruction).cpu()

        # normalize volumes
        original_audio /= original_audio.max()
        reconstructed_audio /= reconstructed_audio.max()

        # send to tensorboard
        self.logger.experiment.add_audio(
            f"Originals/{step}",
            original_audio.clip(-1, 1),
            self.current_epoch
        )
        self.logger.experiment.add_audio(
            f"Reconstructions/{step}",
            reconstructed_audio.clip(-1, 1),
            self.current_epoch
        )

        # save file on disk
        audio_data = torch.cat(
            (
                original_audio,
                torch.zeros(original_audio.size(0), rate),  # 1 second of silence
                reconstructed_audio
            ),
            dim=1
        )
        torchaudio.save(
            f"{self.logger.root_dir}/audio/{step}-{self.current_epoch:02d}.wav",
            audio_data,
            rate
        )
