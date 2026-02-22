from lightning import seed_everything
from pedros import get_logger

from app.data_module import DataModule
from app.logging_config import configure_logging
from app.mnist_module import MNISTModule
from app.settings import get_settings
from app.trainer import get_trainer


class LightningManager:
    def __init__(self):
        configure_logging()
        self.settings = get_settings()
        self.logger = get_logger()

    def start_training(self) -> None:
        seed_everything(self.settings.seed, workers=True)

        data_module = DataModule(self.settings)
        model = MNISTModule(self.settings)
        trainer = get_trainer(self.settings)

        trainer.fit(model, data_module)
        trainer.test(model, data_module)
