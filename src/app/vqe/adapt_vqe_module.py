from collections.abc import Callable
from typing import Literal

import numpy as np
import pennylane as qml
import torch
from scipy.optimize import minimize
from pedros import progbar

from app.settings import AppSettings
from app.vqe.base_module import BaseVQEModule

SingleCandidate = tuple[str, tuple[int, ...]]
DoubleCandidate = tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]
CandidateOp = SingleCandidate | DoubleCandidate
AdaptMode = Literal["standard", "k-adapt", "k-greedy"]


class AdaptVQEModule(BaseVQEModule):
    def __init__(self, settings: AppSettings):
        super().__init__(settings)
        self.automatic_optimization = False

        self._energy_qnodes: dict[tuple[CandidateOp, ...], Callable] = {}
        self._objective_eval_count = 0

        self.adapt_vqe_energy_history: list[float] = []
        self.all_cost_func_history: list[float] = []
        self.adapt_step_plot_indices: list[int] = []
        self.parameter_history: list[np.ndarray] = []

        self.candidate_pool = self._build_candidate_pool()
        self.selected_ops, init_params = self._run_adapt_loop()

        self.theta = torch.nn.Parameter(
            torch.tensor(init_params, dtype=torch.float64), requires_grad=False
        )
        self._circuit = self._get_energy_qnode(self.selected_ops)

    def _resolve_mode_and_k(self) -> tuple[AdaptMode, int]:
        adapt_cfg = self.settings.vqe.adapt
        mode = adapt_cfg.mode

        if mode == "standard":
            return mode, 1

        k = adapt_cfg.k
        if k <= 0:
            raise ValueError("For 'k-adapt' and 'k-greedy', `vqe.adapt.k` must be > 0.")

        return mode, k

    def _build_candidate_pool(self) -> list[CandidateOp]:
        adapt_cfg = self.settings.vqe.adapt
        excitation_level = adapt_cfg.excitation_level

        include_singles = "s" in excitation_level
        include_doubles = "d" in excitation_level

        if "t" in excitation_level:
            self.logger_.warning(
                "`vqe.adapt.excitation_level='sdt'` requested but PennyLane does "
                "not provide fermionic triple excitations in this implementation. "
                "Using singles+doubles only."
            )

        singles, doubles = qml.qchem.excitations(self.n_electrons, self.n_qubits)
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

        pool: list[CandidateOp] = []
        if include_singles:
            for sw in s_wires:
                pool.append(("single", tuple(int(w) for w in sw)))

        if include_doubles:
            for dw in d_wires:
                wires1 = tuple(int(w) for w in dw[0])
                wires2 = tuple(int(w) for w in dw[1])
                pool.append(("double", (wires1, wires2)))

        if not pool:
            raise ValueError(
                "No excitation operators generated. Check `vqe.adapt.excitation_level` "
                "and molecule active-space settings."
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

    def _cost_func(
        self,
        params: np.ndarray,
        ops: list[CandidateOp],
        fixed_params: np.ndarray | None = None,
    ) -> float:
        if fixed_params is not None:
            all_params = np.concatenate((fixed_params, params))
        else:
            all_params = params

        energy = self._energy_for_ops(ops, all_params)

        self._objective_eval_count += 1
        self.all_cost_func_history.append(energy)

        return energy

    def _select_operators(
        self,
        selected_ops: list[CandidateOp],
        selected_params: np.ndarray,
        candidates: list[CandidateOp],
        k: int,
    ) -> list[tuple[float, CandidateOp]]:
        adapt_cfg = self.settings.vqe.adapt
        eps = adapt_cfg.finite_diff_eps

        gradients: list[tuple[float, CandidateOp]] = []
        for candidate_op in candidates:
            ops = [*selected_ops, candidate_op]
            params_plus = np.concatenate([selected_params, np.array([eps])])
            params_minus = np.concatenate([selected_params, np.array([-eps])])

            e_plus = self._energy_for_ops(ops, params_plus)
            e_minus = self._energy_for_ops(ops, params_minus)
            grad = abs((e_plus - e_minus) / (2.0 * eps))
            gradients.append((grad, candidate_op))

        gradients.sort(key=lambda item: item[0], reverse=True)
        return gradients[:k]

    def _optimize_vqe(
        self,
        ansatz_ops: list[CandidateOp],
        current_params: np.ndarray,
        k: int,
        mode: AdaptMode,
    ) -> tuple[np.ndarray, float]:
        adapt_cfg = self.settings.vqe.adapt

        if mode == "k-greedy":
            initial_params = np.zeros(k, dtype=float)
            result = minimize(
                self._cost_func,
                x0=initial_params,
                args=(ansatz_ops, current_params),
                method=adapt_cfg.optimizer_method,
                options={"maxiter": adapt_cfg.max_vqe_iter},
            )
            updated_params = np.concatenate((current_params, np.asarray(result.x)))
        else:
            initial_params = np.append(current_params, np.zeros(k, dtype=float))
            self._objective_eval_count = 0
            result = minimize(
                self._cost_func,
                x0=initial_params,
                args=(ansatz_ops, None),
                method=adapt_cfg.optimizer_method,
                options={"maxiter": adapt_cfg.max_vqe_iter},
            )
            updated_params = np.asarray(result.x)

        current_vqe_energy = float(result.fun)
        self.adapt_vqe_energy_history.append(current_vqe_energy)
        self.adapt_step_plot_indices.append(len(self.all_cost_func_history) - 1)

        return updated_params, current_vqe_energy

    def _run_adapt_loop(self) -> tuple[list[CandidateOp], np.ndarray]:
        adapt_cfg = self.settings.vqe.adapt
        mode, k = self._resolve_mode_and_k()

        selected_ops: list[CandidateOp] = []
        current_optimized_params = np.array([], dtype=float)

        initial_energy = self._energy_for_ops(selected_ops, current_optimized_params)
        self.adapt_vqe_energy_history.append(initial_energy)
        self.all_cost_func_history.append(initial_energy)
        self.adapt_step_plot_indices.append(0)
        self.parameter_history.append(current_optimized_params.copy())

        remaining_pool = list(self.candidate_pool)

        self.logger_.info(
            "Starting ADAPT loop during module initialization "
            f"(mode={mode}, k={k}, max_iters={adapt_cfg.max_adapt_iterations}, "
            f"pool_size={len(self.candidate_pool)})."
        )

        adapt_range = progbar(
            range(adapt_cfg.max_adapt_iterations),
            description="ADAPT iterations",
            total=adapt_cfg.max_adapt_iterations,
        )

        for _ in adapt_range:
            current_pool = (
                remaining_pool if adapt_cfg.drain_pool else self.candidate_pool
            )
            if not current_pool:
                self.logger_.info("ADAPT stopped: operator pool is empty.")
                break

            top_k_candidates = self._select_operators(
                selected_ops,
                current_optimized_params,
                current_pool,
                k,
            )

            if not top_k_candidates:
                self.logger_.info(
                    "ADAPT stopped: no candidate operators were selected."
                )
                break

            max_grad = top_k_candidates[0][0]
            if max_grad < adapt_cfg.gradient_threshold:
                self.logger_.info(
                    "ADAPT converged: "
                    f"max gradient {max_grad:.6e} is below threshold "
                    f"{adapt_cfg.gradient_threshold:.6e}."
                )
                break

            new_operators = [op for _, op in top_k_candidates]
            selected_ops.extend(new_operators)

            current_optimized_params, _ = self._optimize_vqe(
                selected_ops,
                current_optimized_params,
                len(new_operators),
                mode,
            )
            self.parameter_history.append(current_optimized_params.copy())

            if adapt_cfg.drain_pool:
                for op in new_operators:
                    if op in remaining_pool:
                        remaining_pool.remove(op)

        self.logger_.info(
            f"ADAPT initialization finished with {len(selected_ops)} selected operators."
        )
        return selected_ops, current_optimized_params
