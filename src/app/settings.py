from functools import cache
from pathlib import Path
from typing import Any

from pedros import has_dep, get_logger
from pydantic import BaseModel, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings


class DataCfg(BaseModel):
    batch_size: PositiveInt = 64
    num_workers: PositiveInt = 7


class TrainCfg(BaseModel):
    force_cpu: bool = True
    learning_rate: PositiveFloat = 0.001
    max_epochs: PositiveInt = 10
    fast_dev_run: bool = False
    weight_decay: float = 0.0
    callback_verbose: bool = False


class WandbCfg(BaseModel):
    root_path: Path = Path(__file__).resolve().parents[2]

    use_wandb: bool = False
    project: str = "lightning_template"
    entity: str = "deldrel"


class AppSettings(BaseSettings):
    seed: int = 2002

    data: DataCfg = DataCfg()
    train: TrainCfg = TrainCfg()
    wandb: WandbCfg = WandbCfg()

    def __init__(self, **values: Any):
        super().__init__(**values)
        logger = get_logger()
        if self.wandb.use_wandb and not has_dep("wandb"):
            logger.warning(
                "Wandb is enabled in settings but wandb package is not installed. If you want to use it, make sure to add it to the environment with `pip install wandb`."
            )


@cache
def get_settings() -> AppSettings:
    return AppSettings()
