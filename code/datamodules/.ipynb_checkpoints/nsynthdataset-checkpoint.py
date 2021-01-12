import torch.utils.data as data

import glob
import os
import json

class NSynthDataset(data.Dataset):
    def __init__(self, data_dir, **dl_kwargs):
        self.data_dir = data_dir
        self.dict = json.load(open(self.data_dir + "examples.json"))
        self.filenames = list(self.dict.keys())
        
    def filter_instrument(self,inst_str):
        new_dict = self.dict.copy()
        for key in self.dict.keys():
            if inst_str not in self.dict[key]["instrument_str"]:
                new_dict.pop(key)
        self.dict = new_dict.copy()
        self.filenames = list(self.dict.keys())
    
    def __getitem__(self, index):
        name = self.filenames[index]
        sample, _ = torchaudio.load(self.data_dir + 'audio/{}.wav'.format(name))

        Nwin = 512
        spec = torchaudio.functional.spectrogram(sample.view(-1),0,torch.hann_window(Nwin),Nwin,int(Nwin/2),Nwin,None,True)
        spec = torch.sqrt(torch.sum(spec**2,axis=2)) #valeur absolue du spectrogramme
        
        return spec.unsqueeze(0)
        
    def __len__(self):
        return len(self.filenames)