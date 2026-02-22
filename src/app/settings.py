from functools import cache
from pathlib import Path
from typing import Any, Literal

from pedros import get_logger, has_dep
from pydantic import BaseModel, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings


class MoleculeFilesCfg(BaseModel):
    molecules_dir: Path = Path(__file__).resolve().parents[2] / "molecules"
    molecule_id: str = "LiH"


class AdaptCfg(BaseModel):
    enabled: bool = True
    drain_pool: bool = True
    grad_tol: float = 3e-3
    max_steps: PositiveInt = 50


class VQECfg(BaseModel):
    device_name: str = "lightning.qubit"
    molecule: MoleculeFilesCfg = MoleculeFilesCfg()
    adapt: AdaptCfg = AdaptCfg()
    lbfgs_max_iter: PositiveInt = 1


class TrainCfg(BaseModel):
    force_cpu: bool = True
    max_epochs: PositiveInt = 1
    steps_per_epoch: PositiveInt = 100
    fast_dev_run: bool = False
    callback_verbose: bool = False
    learning_rate: PositiveFloat = 0.05
    optimizer: Literal["lbfgs"] = "lbfgs"


class WandbCfg(BaseModel):
    root_path: Path = Path(__file__).resolve().parents[2]
    use_wandb: bool = True
    project: str = "lightning_template_vqe"
    entity: str = "deldrel"


class AppSettings(BaseSettings):
    seed: int = 2002
    train: TrainCfg = TrainCfg()
    vqe: VQECfg = VQECfg()
    wandb: WandbCfg = WandbCfg()

    def __init__(self, **values: Any):
        super().__init__(**values)
        logger = get_logger()
        if self.wandb.use_wandb and not has_dep("wandb"):
            logger.warning(
                "Wandb is enabled in settings but wandb package is not installed. "
                "Add it with `uv add wandb`."
            )


@cache
def get_settings() -> AppSettings:
    return AppSettings()
