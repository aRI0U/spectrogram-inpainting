import torch
from torchvision.utils import make_grid

import model.encoders as encoders
import model.decoders as decoders
from model.quantizers import VectorQuantizer
from model.vqvae.base_vqvae import BaseVQVAE


class MNISTVQVAE(BaseVQVAE):
    def __init__(self,
                 z_dim,
                 num_codewords,
                 commitment_cost,
                 **optimizer_kwargs):
        r"""

        Parameters
        ----------
        z_dim (int): dimension of the latent space
        num_codewords (int): number of codewords
        commitment_cost (float): scaling for commitment loss
        optimizer_kwargs (dict)
        """
        super(MNISTVQVAE, self).__init__(z_dim,
                                         num_codewords,
                                         commitment_cost,
                                         **optimizer_kwargs)

        self.encoder = encoders.MNISTEncoder(1, z_dim)
        self.decoder = decoders.MNISTDecoder(z_dim, 1)

        self.quantizer = VectorQuantizer(num_codewords, z_dim, commitment_cost)

        # for graph logging
        self.example_input_array = torch.randn(1, 1, 32, 32)

    def forward(self, x):
        r"""Forward pass of VQ-VAE

        Parameters
        ----------
        x (torch.FloatTensor): input, shape (B, C, H, W)

        Returns
        -------
        x_hat (torch.FloatTensor): reconstructed input, shape (B, C, H, W)
        codes (torch.LongTensor): corresponding codes, shape (???)
        q_loss (torch.FloatTensor): quantization loss, shape (1)
        """
        # 1. encode and put channels last
        z_e = self.encoder(x).permute(0, 2, 3, 1).contiguous()

        # 2. quantize
        z_q, codes, q_loss = self.quantizer(z_e)

        # 3. put channels first and decode
        x_hat = self.decoder(z_q.permute(0, 3, 1, 2))

        return x_hat, codes, q_loss

    def training_epoch_end(self, outputs):
        x, _ = next(iter(self.train_dataloader()))
        x_hat, codes, _ = self(x)

        # display images in tensorboard
        self.logger.experiment.add_image("Originals/Training", make_grid(x.cpu().data), self.current_epoch)
        self.logger.experiment.add_image("Reconstructions/Training", make_grid(x_hat.cpu().data), self.current_epoch)
        self.plot_codebook_usage(codes, training=True)

    def validation_epoch_end(self, outputs):
        x, _ = next(iter(self.val_dataloader()))
        x_hat, codes, _ = self(x)

        # display images in tensorboard
        self.logger.experiment.add_image("Originals/Validation", make_grid(x.cpu().data), self.current_epoch)
        self.logger.experiment.add_image("Reconstructions/Validation", make_grid(x_hat.cpu().data), self.current_epoch)
        self.plot_codebook_usage(codes, training=False)

        # log hyperparameters
        metrics_log = {'val_loss': torch.stack(outputs).mean()}
        self.logger.log_hyperparams(self.hparams, metrics=metrics_log)
