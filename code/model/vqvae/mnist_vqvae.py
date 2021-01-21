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
