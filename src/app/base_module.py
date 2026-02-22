from abc import abstractmethod, ABC
from typing import Any

import torch
from lightning import LightningModule
from torch.optim import Adam

from app.settings import AppSettings


class BaseModule(LightningModule, ABC):
    def __init__(self, settings: AppSettings):
        super(BaseModule, self).__init__()
        self.settings = settings
        self.loss_func = self.get_loss_func()
        self.save_hyperparameters(ignore=["settings"])

    @abstractmethod
    def get_loss_func(self):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.settings.train.learning_rate)
        return optimizer

    def _step(self, batch, loss_name: str) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log(loss_name, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        return self._step(batch, "val_loss")

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        return self._step(batch, "test_loss")

    def on_train_epoch_start(self) -> None:
        self.log(
            "learning_rate",
            self.optimizers().param_groups[0]["lr"],
            prog_bar=True,
            logger=True,
        )
