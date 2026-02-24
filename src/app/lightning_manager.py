from lightning import seed_everything
from lightning import LightningModule
from pedros import get_logger

from app.logging_config import configure_logging
from app.settings import AppSettings, get_settings
from app.steps_data_module import StepsDataModule
from app.trainer import get_trainer
from app.vqe.adapt_vqe_module import AdaptVQEModule
from app.vqe.uccsd_vqe_module import UCCSDVQEModule


class LightningManager:
    def __init__(self):
        configure_logging()
        self.settings = get_settings()
        self.logger = get_logger()

    def start_training(self) -> None:
        seed_everything(self.settings.seed, workers=True)

        data_module = StepsDataModule(self.settings)
        model = self._build_vqe_module(self.settings)
        trainer = get_trainer(self.settings)

        trainer.fit(model, data_module)

        energy = float(model().detach().cpu().item())
        self.logger.info(f"Final energy: {energy:.10f}")

    @staticmethod
    def _build_vqe_module(settings: AppSettings) -> LightningModule:
        algorithm = settings.vqe.algorithm

        if algorithm == "uccsd":
            return UCCSDVQEModule(settings)
        if algorithm == "adapt":
            return AdaptVQEModule(settings)

        raise ValueError(f"Unsupported VQE algorithm: {algorithm}")
