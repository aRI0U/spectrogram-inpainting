import json

import torchaudio.transforms
from torch.utils.data import Dataset


class NSynthDataset(Dataset):
    def __init__(self, data_dir, nfft, win_length, normalize_spectrograms=False):
        r"""

        Parameters
        ----------
        data_dir (pathlib.Path):
        nfft (int):
        win_length (int):
        normalize_spectrograms (bool):
        """
        self.data_dir = data_dir

        self.transform = torchaudio.transforms.Spectrogram(
            n_fft=nfft,
            win_length=win_length,
            normalized=normalize_spectrograms
        )

        with open(self.data_dir / "examples.json") as f:
            self.dict = json.load(f)

        self.filenames = list(self.dict.keys())

    def filter_instrument(self, inst_str):
        new_dict = self.dict.copy()
        for key in self.dict.keys():
            if inst_str not in self.dict[key]["instrument_str"]:
                new_dict.pop(key)
        self.dict = new_dict.copy()
        self.filenames = list(self.dict.keys())

    def __getitem__(self, index):
        name = self.filenames[index]
        sample, _ = torchaudio.load(self.data_dir / 'audio/{}.wav'.format(name))

        return self.transform(sample)

    def __len__(self):
        return len(self.filenames)
