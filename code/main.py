from datetime import datetime
from pathlib import Path
import sys

# magic trick to make it work on ircam machines
sys.path.insert(1, '/fast-1/alain-atiam/site-packages')

import torch

# https://github.com/pytorch/audio/issues/903
import torchaudio
if sys.platform.startswith("win"):
    torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
else:
    torchaudio.set_audio_backend("sox_io")


import pytorch_lightning as pl

import datamodules
import model.vqvae as vqvae
from utils.parser import Parser
from utils.progress import ProgressBar

parser = Parser()
args = parser.parse_args()

# %% DEBUG
if args.debug:
    # reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # traceback of inf and nan in gradients
    torch.autograd.set_detect_anomaly(True)

# %% DATASET
# loading dataset using Pytorch Lightning datamodules
dm = None
data_dir = Path('datasets') / args.dataset_name
data_dir.mkdir(exist_ok=True, parents=True)

if args.dataset_name == "MNIST":
    dm = datamodules.MNISTDataModule(data_dir, **args.dl_kwargs)

elif args.dataset_name == "NSynth":
    dm = datamodules.NSynthDataModule(
        data_dir,
        args.nfft,
        args.win_length,
        normalize_spectrograms=args.normalize,
        **args.dl_kwargs
    )

else:
    raise NotImplementedError(f"No implementation is provided for this dataset: {args.dataset_name}")

# %% MODEL
# define model
model = None

if args.architecture == 'mnist':
    model = vqvae.MNISTVQVAE(
        args.latent_dim,
        args.num_codewords,
        args.commitment_cost,
        args.gpus,
        **args.adam
    )
else:
    model = vqvae.NSynthVQVAE(
        args.architecture,
        nfft=args.nfft,
        win_length=args.win_length,
        z_dim=args.latent_dim,
        num_codewords=args.num_codewords,
        commitment_cost=args.commitment_cost,
        codebook_restart=args.restarts,
        use_ema=args.ema,
        ema_decay=args.ema_decay,
        gpus=args.gpus,
        **args.adam
    )

# eventually load previously trained model
logs_path = Path('logs')
if args.load_model:
    exp_name = args.load_model
else:
    exp_name = args.dataset_name + '/' + datetime.now().strftime('%Y-%m-%d_%H-%M')
    parser.save(args, logs_path / exp_name)

audio_path = logs_path / exp_name / 'audio'
audio_path.mkdir(parents=True, exist_ok=True)

# %% CALLBACKS
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='Loss/Validation',
    dirpath=logs_path / exp_name,
    filename='checkpoint_{epoch:02d}',
    mode='min'
)

# progress bar
progress_callback = ProgressBar()

# initialize TensorBoard logger
logger = pl.loggers.TensorBoardLogger(logs_path, name=exp_name, version=0, log_graph=True)

# initialize Pytorch Lightning trainer
trainer = pl.Trainer(
    gpus=args.gpus,
    callbacks=[checkpoint_callback, progress_callback],
    logger=logger,
    max_epochs=args.max_epochs
)

# fit and test the model
trainer.fit(model, dm)
trainer.test(datamodule=dm)
