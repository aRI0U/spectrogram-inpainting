from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl

from model.vqvae import VQVAE
from utils.parser import Parser
from utils.progress import ProgressBar

parser = Parser()
args = parser.parse_args()

# loading dataset using Pytorch Lightning datamodules
dm = None
data_dir = Path('datasets') / args.dataset_name
data_dir.mkdir(exist_ok=True, parents=True)

if args.dataset_name == "MNIST":
    from data.mnist import MNISTDataModule

    dm = MNISTDataModule(data_dir, **args.dl_kwargs)

else:
    raise NotImplementedError("No implementation is provided for this dataset")

# define model
model = VQVAE(1, 8, 10, args.commitment_cost)

# eventually load previously trained model
logs_path = Path('logs')
if args.load_model:
    exp_name = args.load_model
else:
    exp_name = args.dataset_name + '/' + datetime.now().strftime('%Y-%m-%d_%H-%M')

# %% CALLBACKS
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='Loss [val]',
    dirpath=logs_path / exp_name,
    filename='checkpoint_{epoch:02d}',
    mode='min'
)

progress_callback = ProgressBar()

# initialize TensorBoard logger
logger = pl.loggers.TensorBoardLogger(logs_path, name=exp_name, version=0)

# initialize Pytorch Lightning trainer
trainer = pl.Trainer(
    callbacks=[checkpoint_callback, progress_callback],
    logger=logger,
    max_epochs=args.max_epochs
)

# fit and test the model
trainer.fit(model, dm)
trainer.test(datamodule=dm)
