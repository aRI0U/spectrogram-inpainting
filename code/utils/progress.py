from tqdm import tqdm

from pytorch_lightning.callbacks import ProgressBar as PlProgressBar


class ProgressBar(PlProgressBar):
    def init_validation_tqdm(self):
        return tqdm(disable=True, leave=False)
