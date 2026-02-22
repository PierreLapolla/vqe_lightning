from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pennylane as qml
import torch
from lightning import LightningModule
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

from app.chem.hamiltonians import build_molecular_hamiltonian
from app.chem.molecule_specs import load_molecule_spec
from app.settings import AppSettings

SingleCandidate = tuple[str, tuple[int, ...]]
DoubleCandidate = tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]
CandidateOp = SingleCandidate | DoubleCandidate


class AdaptVQEModule(LightningModule):
    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self.save_hyperparameters(ignore=["settings"])

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
        self._energy_qnodes: dict[tuple[CandidateOp, ...], Callable] = {}

        self.candidate_pool = self._build_candidate_pool()
        self.selected_ops, init_params = self._build_adapt_ansatz()

        if len(self.selected_ops) == 0:
            raise ValueError(
                "ADAPT selected no operators. Try lowering `vqe.adapt.grad_tol` "
                "or increasing `vqe.adapt.max_steps`."
            )

        self.theta = torch.nn.Parameter(torch.tensor(init_params, dtype=torch.float64))
        self.lbfgs_max_iter = common_cfg.lbfgs_max_iter

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def circuit(theta):
            qml.BasisState(self.hf_state, wires=range(self.n_qubits))
            self._apply_ops(self.selected_ops, theta)
            return qml.expval(self.H)

        self._circuit = circuit

    def _compute_ground_energy(self) -> float:
        sparse_h = self.H.sparse_matrix(wire_order=range(self.n_qubits))
        eigvals = eigsh(sparse_h, k=1, which="SA", return_eigenvectors=False)
        return float(np.real(eigvals[0]))

    def _build_candidate_pool(self) -> list[CandidateOp]:
        singles, doubles = qml.qchem.excitations(self.n_electrons, self.n_qubits)
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

        pool: list[CandidateOp] = []
        for sw in s_wires:
            pool.append(("single", tuple(int(w) for w in sw)))
        for dw in d_wires:
            wires1 = tuple(int(w) for w in dw[0])
            wires2 = tuple(int(w) for w in dw[1])
            pool.append(("double", (wires1, wires2)))

        if not pool:
            raise ValueError(
                "No excitation operators generated. Check active_electrons/"
                "active_orbitals and molecule settings."
            )
        return pool

    def _apply_ops(self, ops: list[CandidateOp], theta) -> None:
        for i, (kind, wires) in enumerate(ops):
            weight = theta[i]
            if kind == "single":
                qml.FermionicSingleExcitation(weight, wires=list(wires))
            else:
                wires1, wires2 = wires
                qml.FermionicDoubleExcitation(
                    weight, wires1=list(wires1), wires2=list(wires2)
                )

    def _get_energy_qnode(self, ops: list[CandidateOp]):
        key = tuple(ops)
        if key in self._energy_qnodes:
            return self._energy_qnodes[key]

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def circuit(theta):
            qml.BasisState(self.hf_state, wires=range(self.n_qubits))
            self._apply_ops(ops, theta)
            return qml.expval(self.H)

        self._energy_qnodes[key] = circuit
        return circuit

    def _energy_for_ops(self, ops: list[CandidateOp], params: np.ndarray) -> float:
        qnode = self._get_energy_qnode(ops)
        theta = torch.tensor(params, dtype=torch.float64)
        return float(qnode(theta).detach().cpu().item())

    def _optimize_params_for_ops(
        self, ops: list[CandidateOp], init_params: np.ndarray
    ) -> np.ndarray:
        if len(ops) == 0:
            return np.array([], dtype=float)

        adapt_cfg = self.settings.vqe.adapt

        def objective(x: np.ndarray) -> float:
            return self._energy_for_ops(ops, x)

        result = minimize(
            objective,
            x0=init_params,
            method="L-BFGS-B",
            options={"maxiter": adapt_cfg.pretrain_maxiter},
        )

        return np.asarray(result.x, dtype=float)

    def _select_next_candidate(
        self,
        selected_ops: list[CandidateOp],
        selected_params: np.ndarray,
        candidates: list[CandidateOp],
    ) -> tuple[CandidateOp | None, float]:
        adapt_cfg = self.settings.vqe.adapt
        eps = adapt_cfg.finite_diff_eps
        if eps <= 0:
            raise ValueError("`vqe.adapt.finite_diff_eps` must be > 0.")

        best_candidate = None
        best_grad = -1.0

        for candidate in candidates:
            ops = [*selected_ops, candidate]
            params_plus = np.concatenate([selected_params, np.array([eps])])
            params_minus = np.concatenate([selected_params, np.array([-eps])])
            e_plus = self._energy_for_ops(ops, params_plus)
            e_minus = self._energy_for_ops(ops, params_minus)
            grad = abs((e_plus - e_minus) / (2.0 * eps))
            if grad > best_grad:
                best_grad = grad
                best_candidate = candidate

        return best_candidate, best_grad

    def _build_adapt_ansatz(self) -> tuple[list[CandidateOp], np.ndarray]:
        adapt_cfg = self.settings.vqe.adapt
        selected_ops: list[CandidateOp] = []
        selected_params = np.array([], dtype=float)
        remaining = list(self.candidate_pool)

        for _ in range(adapt_cfg.max_steps):
            if not remaining:
                break

            candidate, grad = self._select_next_candidate(
                selected_ops, selected_params, remaining
            )
            if candidate is None:
                break

            if grad < adapt_cfg.grad_tol and selected_ops:
                break

            selected_ops.append(candidate)
            selected_params = np.concatenate([selected_params, np.array([0.0])])
            selected_params = self._optimize_params_for_ops(
                selected_ops, selected_params
            )

            if adapt_cfg.drain_pool:
                remaining.remove(candidate)

        return selected_ops, selected_params

    def forward(self) -> torch.Tensor:
        return self._circuit(self.theta)

    def training_step(self, batch, batch_idx):
        loss = torch.abs(self() - self.ground_energy_tensor)
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
