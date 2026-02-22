import numpy as np
import pennylane as qml
import torch
from lightning import LightningModule
from scipy.sparse.linalg import eigsh

from app.settings import AppSettings
from app.chem.molecule_specs import load_molecule_spec
from app.chem.hamiltonians import build_molecular_hamiltonian


class VQEModule(LightningModule):
    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self.save_hyperparameters(ignore=["settings"])

        cfg = settings.vqe
        self.spec = load_molecule_spec(
            cfg.molecule.molecules_dir, cfg.molecule.molecule_id
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
        # Report gap in mHa for more readable live monitoring.
        self.loss_scale = 1_000.0

        self.dev = qml.device(cfg.device_name, wires=self.n_qubits)

        # --- Build a demo chemistry ansatz: UCCSD ---
        singles, doubles = qml.qchem.excitations(self.n_electrons, self.n_qubits)
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

        self.s_wires = s_wires
        self.d_wires = d_wires

        n_params = len(s_wires) + len(d_wires)
        if n_params == 0:
            raise ValueError(
                "No UCCSD excitations generated (n_params=0). "
                "Check active_electrons/active_orbitals and molecule charge/multiplicity."
            )

        init = 0.01 * torch.randn(n_params, dtype=torch.float64)
        self.theta = torch.nn.Parameter(init)

        self.lbfgs_max_iter = cfg.lbfgs_max_iter

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def circuit(theta):
            # UCCSD includes HF initialization via init_state
            qml.UCCSD(
                weights=theta,
                wires=range(self.n_qubits),
                s_wires=self.s_wires,
                d_wires=self.d_wires,
                init_state=self.hf_state,
            )
            return qml.expval(self.H)

        self._circuit = circuit

    def forward(self) -> torch.Tensor:
        return self._circuit(self.theta)

    def _compute_ground_energy(self) -> float:
        sparse_h = self.H.sparse_matrix(wire_order=range(self.n_qubits))
        eigvals = eigsh(sparse_h, k=1, which="SA", return_eigenvectors=False)
        return float(np.real(eigvals[0]))

    def training_step(self, batch, batch_idx):
        loss = torch.abs(self() - self.ground_energy_tensor) * self.loss_scale
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )
        return loss

    def configure_optimizers(self):
        lr = self.settings.train.learning_rate
        opt_name = self.settings.train.optimizer

        if opt_name == "lbfgs":
            return torch.optim.LBFGS(
                [self.theta],
                lr=lr,
                max_iter=self.lbfgs_max_iter,
                line_search_fn="strong_wolfe",
            )

        raise ValueError(f"Unsupported optimizer: {opt_name}")
