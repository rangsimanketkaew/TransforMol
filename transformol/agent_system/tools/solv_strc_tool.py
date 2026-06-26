"""
LangChain Tool: Solute Structure Prediction in Implicit Solvent (MoleculeMLP)

Usage:
    Input JSON: {"smiles": "CCO", "solvent": "water"}
                {"xyz_text": "...", "solvent": "DMSO", "dielectric": 46.7}
    Output: per-atom displacement vectors (Å) from gas to solvated phase and RMSD

Updates:
    13.02.2026  Initial implementation [Rangsiman Ketkaew]
"""

import sys
import json
import traceback
from pathlib import Path

import numpy as np

TRANSFORMOL_ROOT = Path(__file__).resolve().parent.parent.parent
SOLV_STRC_DIR = str(TRANSFORMOL_ROOT / "solv_strc")


def _ensure_path():
    if SOLV_STRC_DIR in sys.path:
        sys.path.remove(SOLV_STRC_DIR)
    sys.path.insert(0, SOLV_STRC_DIR)


def predict_solute_structure(
    smiles,
    config,
    xyz_text=None,
    solvent="water",
    dielectric=None,
):
    """Predict atomic displacements when *smiles* is placed in *solvent*"""

    if isinstance(smiles, str) and smiles.strip().startswith("{"):
        try:
            data = json.loads(smiles)
            if "smiles" in data:
                smiles = data["smiles"]
            if "solvent" in data:
                solvent = data["solvent"]
            if "xyz_text" in data:
                xyz_text = data["xyz_text"]
            if "dielectric" in data:
                dielectric = data["dielectric"]
        except Exception:
            pass

    _ensure_path()

    try:
        import torch
        from py3_train import MoleculeMLP 
        from py4_visualize_prediction import predict_geometry 
        from py2_create_feature_set import calculate_mol_features 
    except ImportError as exc:
        return f"[SolvStrc] Import error: {exc}"

    from transformol.agent_system.mol_utils import get_dielectric, resolve_solvent
    if dielectric is None:
        dielectric = get_dielectric(solvent)
    solvent_smiles = resolve_solvent(solvent)

    try:
        if xyz_text is not None:
            from transformol.agent_system.mol_utils import xyz_text_to_arrays, symbol_to_atomic_number
            atoms, gas_xyz = xyz_text_to_arrays(xyz_text)
            atomic_numbers = np.array([symbol_to_atomic_number(a) for a in atoms], dtype=np.int32)
        elif smiles is not None:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"[SolvStrc] Invalid SMILES: '{smiles}'"
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            atomic_numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
            gas_xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
        else:
            return "[SolvStrc] Provide 'smiles' or 'xyz_text'."
    except Exception as exc:
        return f"[SolvStrc] Geometry generation failed: {exc}\n{traceback.format_exc()}"

    n_atoms = len(atomic_numbers)

    if config.solv_strc_checkpoint is None or config.solv_strc_metadata is None:
        return (
            "[SolvStrc] No checkpoint / metadata provided (demo mode).\n"
            f"  Molecule: {smiles or 'from XYZ'}  |  Solvent: {solvent} (ε={dielectric:.2f})  |  Atoms: {n_atoms}\n"
            "Set config.solv_strc_checkpoint and config.solv_strc_metadata."
        )

    try:
        import json as _json
        with open(config.solv_strc_metadata) as f:
            metadata = _json.load(f)
        device = config.device
        model = MoleculeMLP(
            input_dim=metadata["n_features"],
            output_dim=metadata["n_targets"],
            hidden_dims=metadata["best_hidden_dims"],
            dropout=metadata["best_dropout"],
        ).to(device)
        ckpt = torch.load(config.solv_strc_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as exc:
        return f"[SolvStrc] Model load failed: {exc}\n{traceback.format_exc()}"

    try:
        scaler_mean = np.array(metadata["scaler_mean"])
        scaler_scale = np.array(metadata["scaler_scale"])
        pred_xyz = predict_geometry(
            model=model, gas_xyz=gas_xyz, atomic_numbers=atomic_numbers,
            dielectric=dielectric, scaler_mean=scaler_mean, scaler_scale=scaler_scale,
            device=device, use_displacement=metadata.get("use_displacement", True),
        )[:n_atoms]
    except Exception as exc:
        return f"[SolvStrc] Prediction failed: {exc}\n{traceback.format_exc()}"

    from transformol.agent_system.mol_utils import atomic_number_to_symbol
    disp = pred_xyz - gas_xyz
    rmsd = float(np.sqrt(np.mean(np.sum(disp ** 2, axis=1))))

    lines = [
        "Solute Structure Prediction in Implicit Solvent",
        f"  Molecule: {smiles or 'from XYZ'}  |  Solvent: {solvent} ({solvent_smiles}, ε={dielectric:.2f})",
        f"  Atoms: {n_atoms}  |  RMSD (gas→sol): {rmsd:.4f} Å",
        "",
        f"  {'Atom':>4}  {'Sym':>3}  {'dX':>8}  {'dY':>8}  {'dZ':>8}",
        "  " + "-" * 40,
    ]
    for i, (z, d) in enumerate(zip(atomic_numbers, disp)):
        sym = atomic_number_to_symbol(int(z))
        lines.append(f"  {i:>4}  {sym:>3}  {d[0]:>8.4f}  {d[1]:>8.4f}  {d[2]:>8.4f}")

    return "\n".join(lines)


def build_solv_strc_tool(config):
    """Return a LangChain StructuredTool wrapping"""
    
    from langchain_core.tools import StructuredTool

    def _run(smiles: str = None, xyz_text: str = None, solvent: str = "water", dielectric: float = None) -> str:
        if not smiles and not xyz_text:
            return "[SolvStrc] Provide 'smiles' or 'xyz_text'."
        return predict_solute_structure(smiles, config, xyz_text=xyz_text, solvent=solvent, dielectric=dielectric)

    return StructuredTool.from_function(
        func=_run,
        name="predict_solute_structure",
        description=(
            "Predicts geometry of a solute in implicit solvent using MoleculeMLP. "
            "Input arguments: 'smiles' (string, optional), 'xyz_text' (string, optional), 'solvent' (string, optional, default 'water'), 'dielectric' (float, optional)."
        ),
    )
