from pathlib import Path

import pytorch_lightning as pl

from model.vqvae import VQVAE
from utils.parser import Parser


parser = Parser()
args = parser.parse_args()

trainer = pl.Trainer(max_epochs=10)

dm = None
data_dir = Path('datasets') / args.dataset_name
data_dir.mkdir(exist_ok=True, parents=True)

if args.dataset_name == "MNIST":
    from data.mnist import MNISTDataModule

    dm = MNISTDataModule(data_dir, **args.dl_kwargs)

else:
    raise NotImplementedError("No implementation is provided for this dataset")

model = VQVAE(1, 8, 10, 0.15)

trainer.fit(model, dm)
trainer.test(datamodule=dm)
