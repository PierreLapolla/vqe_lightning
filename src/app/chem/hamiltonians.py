import numpy as np
import pennylane as qml
from pedros import get_logger

from app.chem.molecule_specs import MoleculeSpec


def _as_int(x) -> int:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return int(arr.item())
    raise TypeError(f"Expected scalar int-like value, got array with shape {arr.shape}")


def build_molecular_hamiltonian(spec: MoleculeSpec):
    logger = get_logger()
    logger.info(f"Building Hamiltonian for molecule {spec.molecule_id} ...")

    mol = qml.qchem.Molecule(
        symbols=spec.symbols,
        coordinates=np.array(spec.coordinates, dtype=float),
        charge=spec.charge,
        mult=spec.multiplicity,
        basis_name=spec.basis_name,
        unit=spec.unit,
        name=spec.molecule_id,
    )

    H, n_qubits = qml.qchem.molecular_hamiltonian(
        mol,
        method=spec.method,
        active_electrons=spec.active_electrons,
        active_orbitals=spec.active_orbitals,
        mapping=spec.mapping,
    )

    n_qubits = _as_int(n_qubits)

    n_electrons = (
        spec.active_electrons if spec.active_electrons is not None else mol.n_electrons
    )
    n_electrons = _as_int(n_electrons)

    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
    hf_state = np.array(hf_state, dtype=int)

    logger.info("Done.")

    return H, n_qubits, n_electrons, hf_state
