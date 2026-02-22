from functools import cache
from pathlib import Path
from typing import Any, Literal

from pedros import get_logger, has_dep
from pydantic import BaseModel, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings


class MoleculeFilesCfg(BaseModel):
    molecules_dir: Path = Path(__file__).resolve().parents[2] / "molecules"
    molecule_id: str = "LiH"


class VQECommonCfg(BaseModel):
    device_name: str = "lightning.qubit"
    molecule: MoleculeFilesCfg = MoleculeFilesCfg()
    lbfgs_max_iter: PositiveInt = 10


class UCCSDCfg(BaseModel):
    init_theta_scale: PositiveFloat = 0.01


class AdaptCfg(BaseModel):
    drain_pool: bool = True
    grad_tol: float = 3e-3
    max_steps: PositiveInt = 1
    finite_diff_eps: PositiveFloat = 1e-3
    pretrain_maxiter: PositiveInt = 30


class VQECfg(BaseModel):
    algorithm: Literal["uccsd", "adapt"] = "adapt"
    common: VQECommonCfg = VQECommonCfg()
    uccsd: UCCSDCfg = UCCSDCfg()
    adapt: AdaptCfg = AdaptCfg()


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
