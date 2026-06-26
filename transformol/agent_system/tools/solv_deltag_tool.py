"""
LangChain Tool: Solvation Gibbs Free Energy Prediction (R2S2-GAT)

Usage:
    Input JSON: {"solute_smiles": "CC", "solvent": "water"}
    Output: predicted Gibbs free energy in kcal/mol

Updates:
    13.02.2026  Initial implementation [Rangsiman Ketkaew]
"""

import sys
import json
import traceback
from pathlib import Path

TRANSFORMOL_ROOT = Path(__file__).resolve().parent.parent.parent
SOLV_DELTAG_DIR = str(TRANSFORMOL_ROOT / "solv_deltaG")


def _ensure_path():
    if SOLV_DELTAG_DIR in sys.path:
        sys.path.remove(SOLV_DELTAG_DIR)
    sys.path.insert(0, SOLV_DELTAG_DIR)


def predict_solvation_free_energy(
    solute_smiles,
    solvent,
    config,
):
    """Predict solvation Gibbs free energy (kcal/mol) for *solute_smiles* in *solvent*"""

    if isinstance(solute_smiles, str) and solute_smiles.strip().startswith("{"):
        try:
            data = json.loads(solute_smiles)
            if "solute_smiles" in data:
                solute_smiles = data["solute_smiles"]
            if "solvent" in data:
                solvent = data["solvent"]
        except Exception:
            pass

    _ensure_path()

    try:
        import torch
        from r2s2_train import R2S2GATModel
        from r2s2_dataset import smiles_to_graph
    except ImportError as exc:
        return f"[SolvationGibbs free energy] Import error: {exc}"

    from transformol.agent_system.mol_utils import resolve_solvent
    solvent_smiles = resolve_solvent(solvent)

    try:
        sol = smiles_to_graph(solute_smiles, atom_dim=config.solv_deltag_atom_dim, bond_dim=config.solv_deltag_bond_dim)
        solv = smiles_to_graph(solvent_smiles, atom_dim=config.solv_deltag_atom_dim, bond_dim=config.solv_deltag_bond_dim)
    except Exception as exc:
        return f"[SolvationGibbs free energy] Featurization failed for '{solute_smiles}' / '{solvent_smiles}': {exc}"

    device = config.device
    model = R2S2GATModel(atom_in=config.solv_deltag_atom_dim, edge_in=config.solv_deltag_bond_dim, device=device).to(device)

    if config.solv_deltag_checkpoint is None:
        return (
            "[SolvationGibbs free energy] No checkpoint provided (demo mode).\n"
            f"  Solute: {solute_smiles}  |  Solvent: {solvent} ({solvent_smiles})\n"
            "Set config.solv_deltag_checkpoint to a trained .pt file."
        )

    ckpt_path = Path(config.solv_deltag_checkpoint)
    if not ckpt_path.exists():
        return f"[SolvationGibbs free energy] Checkpoint not found: {ckpt_path}"

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict") or state.get("model_state") or state)

    model.eval()

    try:
        with torch.no_grad():
            preds, _ = model([sol.to(device)], [solv.to(device)])
            delta_g = float(preds[0].cpu().item())
    except Exception as exc:
        return f"[SolvationGibbs free energy] Prediction failed: {exc}\n{traceback.format_exc()}"

    return (
        f"Solvation Free Energy Prediction\n"
        f"  Solute : {solute_smiles}\n"
        f"  Solvent: {solvent} ({solvent_smiles})\n"
        f"  Gibbs free energy_solv: {delta_g:.4f} kcal/mol\n"
    )


def build_solv_deltag_tool(config):
    """Return a LangChain StructuredTool wrapping"""

    from langchain_core.tools import StructuredTool

    def _run(solute_smiles: str, solvent: str = "water") -> str:
        return predict_solvation_free_energy(solute_smiles, solvent, config)

    return StructuredTool.from_function(
        func=_run,
        name="predict_solvation_free_energy",
        description=(
            "Predicts solvation Gibbs free energy (Gibbs free energy, kcal/mol) using R2S2-GAT. "
            "Input arguments: 'solute_smiles' (string, required), 'solvent' (string, optional, default 'water')."
        ),
    )
