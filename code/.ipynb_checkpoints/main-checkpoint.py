from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl

import datamodules
from model.vqvae import VQVAE
from utils.parser import Parser
from utils.progress import ProgressBar

parser = Parser()
args = parser.parse_args()


# %% DATASET
# loading dataset using Pytorch Lightning datamodules
dm = None
data_dir = Path('datasets') / args.dataset_name
data_dir.mkdir(exist_ok=True, parents=True)

if args.dataset_name == "MNIST":
    dm = datamodules.MNISTDataModule(data_dir, **args.dl_kwargs)

elif args.dataset_name == "NSynth":
    dm = datamodules.NSynthDataModule(data_dir, **args.dl_kwargs)

else:
    raise NotImplementedError("No implementation is provided for this dataset")

# %% MODEL
# define model
model = VQVAE(
    args.architecture,
    1,
    args.latent_dim,
    args.num_codewords,
    args.commitment_cost,
    **args.adam
)

# eventually load previously trained model
logs_path = Path('logs')
if args.load_model:
    exp_name = args.load_model
else:
    exp_name = args.dataset_name + '/' + datetime.now().strftime('%Y-%m-%d_%H-%M')
    parser.save(args, logs_path / exp_name)

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
    callbacks=[checkpoint_callback, progress_callback],
    logger=logger,
    max_epochs=args.max_epochs
)

# fit and test the model
trainer.fit(model, dm)
trainer.test(datamodule=dm)