from pathlib import Path
from typing import Optional

from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from app.settings import AppSettings


class DataModule(LightningDataModule):
    def __init__(self, settings: AppSettings) -> None:
        super(DataModule, self).__init__()
        self.settings = settings
        self.data_path = Path(__file__).parent.parent.parent / "data"

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.predict_set = None

    def prepare_data(self) -> None:
        MNIST(self.data_path, train=True, download=True)
        MNIST(self.data_path, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        valid_stages = {None, "fit", "validate", "test", "predict"}
        if stage not in valid_stages:
            message = f"Stage '{stage}' is not recognized."
            raise ValueError(message)

        if stage in (None, "fit", "validate") and (
            self.train_set is None or self.val_set is None
        ):
            mnist_full = MNIST(self.data_path, train=True, transform=ToTensor())
            split_generator = Generator().manual_seed(self.settings.seed)
            self.train_set, self.val_set = random_split(
                mnist_full,
                [55000, 5000],
                generator=split_generator,
            )

        if stage in (None, "test") and self.test_set is None:
            self.test_set = MNIST(self.data_path, train=False, transform=ToTensor())

        if stage in (None, "predict") and self.predict_set is None:
            self.predict_set = MNIST(self.data_path, train=False, transform=ToTensor())

    def _get_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.settings.data.batch_size,
            num_workers=self.settings.data.num_workers,
            persistent_workers=True,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_set)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_set)

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.predict_set)
