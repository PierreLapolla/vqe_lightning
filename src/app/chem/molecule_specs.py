from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class MoleculeSpec:
    molecule_id: str
    symbols: list[str]
    coordinates: np.ndarray
    unit: str
    charge: int
    multiplicity: int
    basis_name: str
    mapping: str
    method: str
    active_electrons: int | None
    active_orbitals: int | None


_REQUIRED_KEYS = {
    "unit",
    "charge",
    "multiplicity",
    "basis_name",
    "mapping",
    "method",
    "active_electrons",
    "active_orbitals",
}


def _read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"Invalid XYZ file: {path}")
    body = lines[2:]
    symbols: list[str] = []
    coords: list[list[float]] = []
    for ln in body:
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ line: {ln}")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.array(coords, dtype=float)


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required. Install with `uv add pyyaml`.")
    return yaml.safe_load(path.read_text()) or {}


def load_molecule_spec(spec_dir: Path, molecule_id: str) -> MoleculeSpec:
    xyz_path = spec_dir / f"{molecule_id}.xyz"
    yml_path = spec_dir / f"{molecule_id}.yaml"

    if not xyz_path.exists():
        raise FileNotFoundError(f"Missing XYZ file: {xyz_path}")
    if not yml_path.exists():
        raise FileNotFoundError(
            f"Missing YAML file: {yml_path}. All chemistry settings must be defined in YAML."
        )

    symbols, coords = _read_xyz(xyz_path)
    meta = _read_yaml(yml_path)

    missing = sorted(_REQUIRED_KEYS - set(meta.keys()))
    if missing:
        raise ValueError(f"Missing required YAML keys in {yml_path}: {missing}")

    return MoleculeSpec(
        molecule_id=molecule_id,
        symbols=symbols,
        coordinates=coords,
        unit=str(meta["unit"]),
        charge=int(meta["charge"]),
        multiplicity=int(meta["multiplicity"]),
        basis_name=str(meta["basis_name"]),
        mapping=str(meta["mapping"]),
        method=str(meta["method"]),
        active_electrons=(
            int(meta["active_electrons"])
            if meta.get("active_electrons") is not None
            else None
        ),
        active_orbitals=(
            int(meta["active_orbitals"])
            if meta.get("active_orbitals") is not None
            else None
        ),
    )
