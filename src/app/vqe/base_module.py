import numpy as np
import pennylane as qml
import torch
from lightning import LightningModule
from pedros import get_logger
from scipy.sparse.linalg import eigsh

from app.chem.hamiltonians import build_molecular_hamiltonian
from app.chem.molecule_specs import load_molecule_spec
from app.settings import AppSettings


class BaseVQEModule(LightningModule):
    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self.save_hyperparameters(ignore=["settings"])
        self.logger_ = get_logger()

        vqe_cfg = settings.vqe
        common_cfg = vqe_cfg.common

        self.spec = load_molecule_spec(
            common_cfg.molecule.molecules_dir, common_cfg.molecule.molecule_id
        )
        self.H, self.n_qubits, self.n_electrons, self.hf_state = (
            build_molecular_hamiltonian(self.spec)
        )

        if not isinstance(self.n_electrons, int):
            raise TypeError(f"n_electrons must be int, got {type(self.n_electrons)}")

        self.ground_energy = self._compute_ground_energy()
        self.register_buffer(
            "ground_energy_tensor",
            torch.tensor(self.ground_energy, dtype=torch.float64),
        )

        self.dev = qml.device(common_cfg.device_name, wires=self.n_qubits)
        self.lbfgs_max_iter = common_cfg.lbfgs_max_iter

    def _compute_ground_energy(self) -> float:
        sparse_h = self.H.sparse_matrix(wire_order=range(self.n_qubits))
        eigvals = eigsh(sparse_h, k=1, which="SA", return_eigenvectors=False)
        return float(np.real(eigvals[0]))

    def training_step(self, batch, batch_idx):
        energy = self()
        abs_delta = torch.abs(energy - self.ground_energy_tensor)

        self.log(
            "energy", energy, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log(
            "energy_abs_delta",
            abs_delta,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        return energy

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            return None

        lr = self.settings.train.learning_rate
        opt_name = self.settings.train.optimizer

        if opt_name == "lbfgs":
            return torch.optim.LBFGS(
                params,
                lr=lr,
                max_iter=self.lbfgs_max_iter,
                line_search_fn="strong_wolfe",
            )

        raise ValueError(f"Unsupported optimizer: {opt_name}")

    def forward(self) -> torch.Tensor:
        return self._circuit(self.theta)
