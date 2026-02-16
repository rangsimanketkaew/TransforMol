"""
Shared molecular utilities for all TransforMol agents

Updates:
    13.02.2026  Initial implementation [Rangsiman Ketkaew]
"""

import sys
from pathlib import Path

import numpy as np

_TRANSFORMOL_ROOT = Path(__file__).resolve().parent.parent


def _ensure_module_on_path(module_dir):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)


SOLVENT_SMILES: dict[str, str] = {
    "water": "O", "h2o": "O",
    "methanol": "CO", "meoh": "CO",
    "ethanol": "CCO", "etoh": "CCO",
    "isopropanol": "CC(C)O", "ipa": "CC(C)O", "2-propanol": "CC(C)O",
    "acetonitrile": "CC#N", "mecn": "CC#N",
    "dmso": "CS(C)=O", "dimethylsulfoxide": "CS(C)=O",
    "dmf": "CN(C)C=O", "dimethylformamide": "CN(C)C=O",
    "thf": "C1CCOC1", "tetrahydrofuran": "C1CCOC1",
    "acetone": "CC(C)=O", "propanone": "CC(C)=O",
    "dioxane": "C1CCOCC1",
    "hexane": "CCCCCC",
    "cyclohexane": "C1CCCCC1",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "diethyl ether": "CCOCC", "ether": "CCOCC",
    "chloroform": "ClC(Cl)Cl",
    "dcm": "ClCCl", "dichloromethane": "ClCCl", "methylene chloride": "ClCCl",
    "ethyl acetate": "CCOC(C)=O", "etoac": "CCOC(C)=O",
    "acetic acid": "CC(O)=O",
    "formic acid": "OC=O",
    "pyridine": "c1ccncc1",
}

SOLVENT_DIELECTRIC: dict[str, float] = {
    "water": 78.4, "h2o": 78.4,
    "methanol": 32.7, "meoh": 32.7,
    "ethanol": 24.5, "etoh": 24.5,
    "isopropanol": 18.3, "ipa": 18.3,
    "acetonitrile": 37.5, "mecn": 37.5,
    "dmso": 46.7, "dimethylsulfoxide": 46.7,
    "dmf": 36.7, "dimethylformamide": 36.7,
    "thf": 7.6, "tetrahydrofuran": 7.6,
    "acetone": 20.7, "propanone": 20.7,
    "hexane": 1.88,
    "cyclohexane": 2.02,
    "benzene": 2.27,
    "toluene": 2.38,
    "diethyl ether": 4.34, "ether": 4.34,
    "chloroform": 4.81,
    "dcm": 8.93, "dichloromethane": 8.93,
    "ethyl acetate": 6.02, "etoac": 6.02,
    "acetic acid": 6.2,
    "pyridine": 12.4,
}

_Z_TO_SYM: dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 35: "Br", 36: "Kr",
    53: "I",
}
_SYM_TO_Z: dict[str, int] = {v: k for k, v in _Z_TO_SYM.items()}


def get_solvent_smiles(name):
    """Return SMILES for a common solvent name (case-insensitive), or None."""
    return SOLVENT_SMILES.get(name.strip().lower())


def get_dielectric(solvent_name, default=78.4):
    """Return dielectric constant for a solvent name; falls back to *default*."""
    return SOLVENT_DIELECTRIC.get(solvent_name.strip().lower(), default)


def resolve_solvent(solvent_input):
    """Return SMILES for *solvent_input* (name or raw SMILES pass-through)."""
    return get_solvent_smiles(solvent_input) or solvent_input


def atomic_number_to_symbol(z):
    return _Z_TO_SYM.get(int(z), f"Z{z}")


def symbol_to_atomic_number(symbol):
    z = _SYM_TO_Z.get(symbol.capitalize())
    if z is None:
        raise ValueError(f"Unknown element symbol: '{symbol}'")
    return z


def xyz_text_to_arrays(xyz_text):
    """Parse an XYZ-format string → (atom symbols, coords array (N,3))."""
    lines = [ln.strip() for ln in xyz_text.strip().splitlines()]
    if len(lines) < 2:
        raise ValueError("XYZ text must have at least 2 lines.")
    try:
        n_atoms = int(lines[0])
    except ValueError:
        raise ValueError(f"First line must be atom count, got: '{lines[0]}'")
    atom_lines = lines[2: 2 + n_atoms]
    if len(atom_lines) < n_atoms:
        raise ValueError(f"Expected {n_atoms} coordinate lines, found {len(atom_lines)}.")
    atoms, coords_list = [], []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ line: '{line}'")
        atoms.append(parts[0])
        coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(coords_list, dtype=np.float32)


def smiles_to_pyg_graph(smiles, atom_dim=30, bond_dim=6):
    """Convert a SMILES string to a PyG Data object (delegates to r2s2_dataset)."""
    solv_dir = str(_TRANSFORMOL_ROOT / "solv_deltaG")
    _ensure_module_on_path(solv_dir)
    from r2s2_dataset import smiles_to_graph  # type: ignore[import]
    return smiles_to_graph(smiles, atom_dim=atom_dim, bond_dim=bond_dim)
