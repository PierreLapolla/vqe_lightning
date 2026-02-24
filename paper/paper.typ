#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [ADAPT-VQE Lightning Pipeline: Inputs, Transformations, and Outputs],
  abstract: [
    This document presents the current ADAPT-VQE Lightning implementation as an explicit process specification. It focuses on three layers: the molecular and runtime inputs, the mathematical and computational transformations applied to those inputs, and the quantitative outputs produced by training and checkpoint artifacts. The objective is to provide a clear and reproducible map of how this codebase turns chemistry specifications into variational energies and error metrics.
  ],
  authors: (
    (
      name: "Thomas Roustan",
      email: ""
    ),
    (
      name: "Pierre Lapolla",
      email: "pro@pierrelapolla.com"
    ),
  ),
  index-terms: ("quantum chemistry", "variational quantum eigensolver", "ADAPT-VQE", "PennyLane", "PyTorch Lightning"),
  bibliography: bibliography("references.bib"),
)

This document is process-oriented and explains three things directly:
- what inputs are provided,
- what transformations are applied,
- what outputs and quantitative results are produced.

The workflow follows the VQE formulation @peruzzo2014 and ADAPT-VQE ansatz growth @grimsley2019, with Jordan-Wigner qubit mapping @jordan1928 and UCC-style fermionic excitations @romero2018.

= Inputs

== Molecular and Chemistry Inputs

The molecule is split across two files:
- `molecules/LiH.xyz` (geometry): Li at `(0, 0, 0)` and H at `(0, 0, 1.5712755877)` Angstrom.
- `molecules/LiH.yaml` (chemistry metadata): `charge = 0`, `multiplicity = 1`, `basis_name = sto-3g`, `mapping = jordan_wigner`, `method = dhf`, `active_electrons = 2`, `active_orbitals = 4`.

The electronic structure target can be summarized as the ground-state problem
$E_0 = min_(psi) psi^dagger H psi$,
where $H$ is the second-quantized molecular Hamiltonian.

== Runtime and Optimization Inputs

Default runtime parameters (from `src/app/settings.py`) are:
- `seed = 2002`
- `vqe.algorithm = "adapt"`
- `vqe.common.device_name = "lightning.qubit"`
- `vqe.common.lbfgs_max_iter = 10`
- `vqe.adapt.max_steps = 1`
- `vqe.adapt.grad_tol = 3e-3`
- `vqe.adapt.finite_diff_eps = 1e-3`
- `vqe.adapt.pretrain_maxiter = 30`
- `train.max_epochs = 1`
- `train.steps_per_epoch = 100`
- `train.optimizer = "lbfgs"`
- `train.learning_rate = 0.05`

= Transformations

== 1) Build the Hamiltonian and Reference State

`src/app/chem/molecule_specs.py` loads and validates molecule data, then `src/app/chem/hamiltonians.py` constructs:
- qubit Hamiltonian `H_q`,
- number of qubits `n`,
- number of active electrons `N_e`,
- Hartree-Fock bitstring `|phi_HF>` used as reference.

The fermionic Hamiltonian has the standard form
$H_f = sum_(p q) h_(p q) a_p^dagger a_q + 1 / 2 sum_(p q r s) h_(p q r s) a_p^dagger a_q^dagger a_r a_s$,
then is mapped (Jordan-Wigner @jordan1928) to
$H_q = sum_(k=1)^M c_k P_k, quad P_k in {I, X, Y, Z}^n$.

The exact target energy is computed by sparse diagonalization (`eigsh`, SciPy @virtanen2020scipy):
$E_"exact" = min_(psi) psi^dagger H_q psi$.

== 2) Build the ADAPT Operator Pool and Select New Operators

`src/app/adapt_vqe_module.py` creates an excitation pool from `qml.qchem.excitations`:
$P = {tau_i}_(i=1)^(N_"pool")$.

At each ADAPT step, each candidate is scored with a finite-difference gradient proxy
$g_i approx (E(theta, +epsilon e_i) - E(theta, -epsilon e_i)) / (2 epsilon)$.

The selected operator is
$i^* = arg max_i |g_i|$,
and the ansatz is extended as
$U^(t+1)(theta) = exp(theta_(t+1) tau_(i^*)) U^(t)(theta)$.

After insertion, parameters are pre-optimized with L-BFGS-B (SciPy) before Lightning training.

== 3) Outer Optimization in Lightning

The model objective is the absolute energy gap to the exact solver:
$L(theta) = |E(theta) - E_"exact"|$.

Training runs over a synthetic step dataset (`src/app/steps_data_module.py`) so optimization length is controlled by `steps_per_epoch`. Checkpoints and metrics are emitted by callbacks configured in `src/app/trainer.py`.

= Outputs

== Static Outputs Produced During Setup

For the default LiH configuration, module construction produces:
- `n_qubits = 8`
- `n_electrons = 2`
- `hf_state = [1, 1, 0, 0, 0, 0, 0, 0]`
- `Hamiltonian terms M = 105`
- Hilbert-space dimension `2^8 = 256`
- ADAPT pool size `N_pool = 15`
- selected first ADAPT operator: `("double", ((0, 1), (4, 5)))`
- strongest initial gradient magnitude: `|g_i| = 0.023603439331054688`

== Training and Artifact Outputs

Produced files include:
- `lightning_logs/version_*/metrics.csv` (per-step logged losses),
- `lightning_logs/version_*/checkpoints/*.ckpt` (model states),
- optional W&B run directories under `wandb/` and `lightning_template_vqe/`.

== Numerical Results from Repository Checkpoints

Reference exact value from the stored checkpoints:
- `E_exact = -7.8644891084057305` Hartree.

Initial (Hartree-Fock / zero-parameter) value for the selected one-operator ADAPT ansatz:
- `E_HF = -7.86266565322876` Hartree,
- `|E_HF - E_exact| = 0.001823455176967137` Hartree.

Checkpointed ADAPT values:
- `lightning_logs/version_0/checkpoints/vqe-step=00002-train_loss=0.001717.ckpt`:
  `theta = 0.004659244527978341`, `E = -7.862771987915039`,
  `|E - E_exact| = 0.001717120490677182` Hartree.
- `lightning_logs/version_1/checkpoints/vqe-step=00003-train_loss=0.001646.ckpt`:
  `theta = 0.00796484071212286`, `E = -7.862843036651611`,
  `|E - E_exact| = 0.0016460717541191272` Hartree.

So, for this current one-step ADAPT setup, optimization improves the energy gap relative to the HF starting point, but does not yet reach chemical-accuracy scale. This is consistent with ADAPT-VQE behavior when ansatz depth is intentionally limited @grimsley2019.

= Process Summary

End-to-end, the pipeline is:
1. Load geometry + chemistry metadata.
2. Build molecular Hamiltonian and HF state.
3. Map to qubits and compute exact reference energy.
4. Build ADAPT pool, score candidates, append best operator.
5. Pre-optimize with L-BFGS-B.
6. Train remaining parameters with Lightning/LBFGS on step-index data.
7. Emit energies, losses, checkpoints, and optional experiment logs.

This format is intended to make future updates straightforward: each new molecule or optimization setting can be documented by updating the same input/process/output slots.
