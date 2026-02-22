from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from app.settings import AppSettings


class _StepsDataset(Dataset):
    def __init__(self, n_steps: int):
        self.n_steps = n_steps

    def __len__(self) -> int:
        return self.n_steps

    def __getitem__(self, idx: int):
        return 0  # unused


class StepsDataModule(LightningDataModule):
    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self._ds = _StepsDataset(settings.train.steps_per_epoch)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._ds, batch_size=1, shuffle=False, num_workers=0)
