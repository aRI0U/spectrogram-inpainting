r"""Wrap NSynth dataset with a Lightning Data Module"""

import requests
from tqdm import tqdm

from torch.utils.data import DataLoader

import pytorch_lightning as pl


class NSynthDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, **dl_kwargs):
        super(NSynthDataModule, self).__init__()

        self.data_dir = data_dir
        self.dl_kwargs = dl_kwargs

        self.nsynth_train = None
        self.nsynth_val = None
        self.nsynth_test = None

    def prepare_data(self):
        r"""Download and extract the NSynth dataset in the folder ./datasets if it is not already downloaded."""
        # TODO: check if dataset already downloaded
        print("WARNING: NSynth dataset is quite huge (voir quelle taille ça fait).", end=" ")
        if input("Download anyway? [yes/no] ") in "yes":
            # TODO: download and extract the dataset
            pass

    def setup(self, stage=None):
        # TODO: initialize train, val and test sets in this function, maybe create an external NSynthDataset class...
        pass

    def train_dataloader(self):
        return DataLoader(self.nsynth_train, **self.dl_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.nsynth_val, **self.dl_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.nsynth_test, **self.dl_kwargs, shuffle=False)

    @staticmethod
    def download(url, filename):
        with open(filename, "wb") as f:
            r = requests.get(url, stream=True)
            pbar = tqdm(unit="kB", total=int(r.headers['Content-Length']) // 1024)
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    pbar.update()
                    f.write(chunk)


if __name__ == '__main__':
    NSynthDataModule.download('http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz', 'tmp.tar.gz')