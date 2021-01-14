import json

import torch
import torchaudio
from torch.utils.data import Dataset


class NSynthDataset(Dataset):
    def __init__(self, data_dir, nfft, win_length):
        r"""

        Parameters
        ----------
        data_dir (pathlib.Path):
        nfft (int):
        win_length (int):
        """
        self.data_dir = data_dir
        self.nfft = nfft
        self.win_length = win_length
        self.window = torch.hann_window(self.win_length)

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

        spec = torchaudio.functional.spectrogram(sample,
                                                 pad=0,
                                                 window=self.window,
                                                 n_fft=self.nfft,
                                                 hop_length=self.win_length // 2,
                                                 win_length=self.win_length,
                                                 power=None,
                                                 normalized=True)
        spec = torch.sqrt(torch.sum(spec ** 2, axis=-1))  # valeur absolue du spectrogramme
        return spec

    def __len__(self):
        return len(self.filenames)
