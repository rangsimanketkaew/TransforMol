"""
LangChain Tool: Reactive Atom Prediction (GNN + Pipek-Mezey localization)

Usage:
    Input JSON: {"smiles": "C1CCCCC1", "xyz_text": "...", "top_k": 5}
    Output: ranked list of atoms by reactivity score

Updates:
    13.02.2026  Initial implementation [Rangsiman Ketkaew]
"""

import sys
import json
import traceback
from pathlib import Path

import numpy as np

TRANSFORMOL_ROOT = Path(__file__).resolve().parent.parent.parent
_REACTIVE_ATOM_DIR = str(TRANSFORMOL_ROOT / "reactive_atom")


def _ensure_path():
    if _REACTIVE_ATOM_DIR in sys.path:
        sys.path.remove(_REACTIVE_ATOM_DIR)
    sys.path.insert(0, _REACTIVE_ATOM_DIR)


def predict_reactive_atoms(
    smiles,
    config,
    xyz_text=None,
    top_k=5,
):
    """Rank atoms in *smiles* by reactivity using the GNN localization model"""

    if isinstance(smiles, str) and smiles.strip().startswith("{"):
        try:
            data = json.loads(smiles)
            if "smiles" in data:
                smiles = data["smiles"]
            if "top_k" in data:
                top_k = data["top_k"]
            if "xyz_text" in data:
                xyz_text = data["xyz_text"]
        except Exception:
            pass

    _ensure_path()
    for mod in ["model", "data", "loss"]:
        sys.modules.pop(mod, None)

    try:
        import torch
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from model import MLWithLocalization, predict
        from data import xyz_to_data
    except ImportError as exc:
        return f"[ReactiveAtom] Import error: {exc}"

    try:
        if xyz_text is not None:
            from transformol.agent_system.mol_utils import xyz_text_to_arrays, symbol_to_atomic_number
            atoms, coords = xyz_text_to_arrays(xyz_text)
            Z = np.array([symbol_to_atomic_number(a) for a in atoms], dtype=np.int32)
            item = xyz_to_data(Z, coords, target=0.0)
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"[ReactiveAtom] Invalid SMILES: '{smiles}'"
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
            item = xyz_to_data(Z, coords, target=0.0)
    except Exception as exc:
        return f"[ReactiveAtom] Graph build failed: {exc}\n{traceback.format_exc()}"

    device = config.device
    model = MLWithLocalization(
        model=config.reactive_atom_model_type,
        hidden_dim=config.reactive_atom_hidden_dim,
        n_orb=config.reactive_atom_n_orb,
        loc_mode="per_atom",
    ).to(device)

    if config.reactive_atom_checkpoint is None:
        return (
            "[ReactiveAtom] No checkpoint provided (demo mode).\n"
            f"  Molecule: {smiles}  |  Atoms: {len(Z)}\n"
            "Set config.reactive_atom_checkpoint to a trained .pt file."
        )

    ckpt_path = Path(config.reactive_atom_checkpoint)
    if not ckpt_path.exists():
        return f"[ReactiveAtom] Checkpoint not found: {ckpt_path}"

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state") or state.get("model_state_dict") or state)

    try:
        results = predict(model, [item], device=device, reactive_mode=config.reactive_atom_mode)
    except Exception as exc:
        return f"[ReactiveAtom] Prediction failed: {exc}\n{traceback.format_exc()}"

    res = results[0]
    ranking = res["reactive_ranking_indices"][:top_k]
    scores = res["atom_scores"]

    from transformol.agent_system.mol_utils import atomic_number_to_symbol
    lines = [
        "Reactive Atom Prediction",
        f"  Molecule: {smiles}  |  Atoms: {len(Z)}  |  Energy: {res['E_pred']:.4f} a.u.",
        f"  Top {top_k} reactive atoms:",
    ]
    for rank, idx in enumerate(ranking, 1):
        sym = atomic_number_to_symbol(int(Z[idx]))
        lines.append(f"    Rank {rank}: atom {idx} ({sym})  score {scores[idx]:.4f}")

    return "\n".join(lines)


def build_reactive_atom_tool(config):
    """Return a LangChain StructuredTool wrapping"""
    
    from langchain_core.tools import StructuredTool

    def _run(smiles: str, top_k: int = 5, xyz_text: str = None) -> str:
        return predict_reactive_atoms(
            smiles, config,
            xyz_text=xyz_text,
            top_k=top_k,
        )

    return StructuredTool.from_function(
        func=_run,
        name="predict_reactive_atoms",
        description=(
            "Ranks atoms by reactivity for transition-state search using GNN + PM localization. "
            "Input arguments: 'smiles' (string, required), 'top_k' (integer, optional, default 5), 'xyz_text' (string, optional)."
        ),
    )
