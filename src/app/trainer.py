from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from pedros import has_dep

from app.settings import AppSettings


def get_trainer(settings: AppSettings) -> Trainer:
    callbacks = []

    callbacks.append(
        ModelCheckpoint(
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_train_steps=1,
            save_on_train_epoch_end=False,
            filename="vqe-{step:05d}-{train_loss:.6f}",
            verbose=settings.train.callback_verbose,
        )
    )

    if has_dep("rich"):
        callbacks.append(RichProgressBar())

    trainer_logger = None
    if settings.wandb.use_wandb and has_dep("wandb"):
        trainer_logger = WandbLogger(
            project=settings.wandb.project,
            entity=settings.wandb.entity,
            save_dir=str(settings.wandb.root_path),
            log_model=True,
        )

    return Trainer(
        accelerator="cpu" if settings.train.force_cpu else "auto",
        max_epochs=settings.train.max_epochs,
        callbacks=callbacks,
        logger=trainer_logger,
        fast_dev_run=settings.train.fast_dev_run,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
