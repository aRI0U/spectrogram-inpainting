r"""Wrap NSynth dataset with a Lightning Data Module"""

import requests
import tarfile
from tqdm import tqdm

from torch.utils.data import DataLoader

import pytorch_lightning as pl

DATASET_URL_FORMAT = 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-{}.jsonwav.tar.gz'
ALL_SUBSETS = ['test', 'valid']

class NSynthDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, **dl_kwargs):
        super(NSynthDataModule, self).__init__()

        self.data_dir = data_dir
        self.dl_kwargs = dl_kwargs

        self.nsynth_train = None
        self.nsynth_val = None
        self.nsynth_test = None

    def prepare_data(self):
        self.download_data()


    def setup(self, stage=None):
        # TODO: initialize train, val and test sets in this function, maybe create an external NSynthDataset class...
        pass

    def train_dataloader(self):
        return DataLoader(self.nsynth_train, **self.dl_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.nsynth_val, **self.dl_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.nsynth_test, **self.dl_kwargs, shuffle=False)

    def download_data(self):
        r"""Download and extract the NSynth dataset
        in the folder self.data_dir if it is not already downloaded."""
        # check whether all subsets are already downloaded or not
        if all(subset in self.data_dir.iterdir() for subset in ALL_SUBSETS):
            return

        # ask which subsets should be downloaded
        print("WARNING: NSynth dataset is quite huge (almost 30 GB).", end=" ")
        subsets_to_dl = input(f"Which subset(s) do you want to download? [{'/'.join(ALL_SUBSETS)}] ").split(' ')

        for set in subsets_to_dl:
            # filter invalid subsets
            if set not in ALL_SUBSETS:
                continue

            # download set
            url = DATASET_URL_FORMAT.format(set)
            filename = self.data_dir / f"nsynth-{set}.tar.gz"
            print(f"Downloading {url}. It might take some time...")
            self.download(url, filename)

            # extract downloaded archive
            if not filename.exists():
                continue
            print(f"Extracting {filename}...")
            tar = tarfile.open(filename, "r:gz")
            tar.extractall(self.data_dir)
            tar.close()

            # rename the output directory
            output_dir = self.data_dir / f"nsynth-{set}"
            output_dir.rename(self.data_dir / set)

            # delete the archive to save space
            filename.unlink()

    @staticmethod
    def download(url, filename):
        r"""Download a file from the web

        Parameters
        ----------
        url (str): URL of the file
        filename (str or pathlib.Path): destination file
        """
        # if filename.exists():
        #     print(f"{filename} already exists. Download aborted.")
        with open(filename, "wb") as f:
            r = requests.get(url, stream=True)
            pbar = tqdm(unit="kB", total=int(r.headers['Content-Length']) // 1024)
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    pbar.update()
                    f.write(chunk)


if __name__ == '__main__':
    NSynthDataModule.download('http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz', 'tmp.tar.gz')