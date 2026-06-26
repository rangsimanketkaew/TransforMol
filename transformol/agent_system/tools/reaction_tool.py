"""
LangChain Tool: Chemical Reaction Prediction (MPNN + CVAE)

Usage:
    Input JSON: {"smiles": "CCO", "num_samples": 3}
    Output: reactive atom scores and TS/product feature norms per hypothesis

Updates:
    13.02.2026  Initial implementation [Rangsiman Ketkaew]
"""

import sys
import json
import traceback
from pathlib import Path

TRANSFORMOL_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICT_REACTION_DIR = str(TRANSFORMOL_ROOT / "predict_reaction")


def _ensure_path():
    if PREDICT_REACTION_DIR not in sys.path:
        sys.path.insert(0, PREDICT_REACTION_DIR)


def predict_reaction(
    reactant_smiles,
    config,
    num_samples=None,
):
    """Predict TS and product structures from *reactant_smiles*"""

    _ensure_path()
    if num_samples is None:
        num_samples = config.reaction_num_samples

    try:
        import torch
        from model import ReactionGenerativeModel 
    except ImportError as exc:
        return f"[Reaction] Import error: {exc}"

    try:
        from transformol.agent_system.mol_utils import smiles_to_pyg_graph
        reactant_data = smiles_to_pyg_graph(reactant_smiles, atom_dim=config.reaction_node_dim, bond_dim=config.reaction_edge_dim)
        reactant_data = reactant_data.to(config.device)
        reactant_data.edge_index = reactant_data.edge_index.to(torch.int64)
    except Exception as exc:
        return f"[Reaction] Featurization failed for '{reactant_smiles}': {exc}"

    device = config.device
    model = ReactionGenerativeModel(
        node_in_dim=config.reaction_node_dim,
        edge_in_dim=config.reaction_edge_dim,
        hidden_dim=config.reaction_hidden_dim,
        latent_dim=config.reaction_latent_dim,
    ).to(device)

    if config.reaction_checkpoint is None:
        return (
            "[Reaction] No checkpoint provided (demo mode).\n"
            f"  Reactant: {reactant_smiles}  |  Samples: {num_samples}\n"
            "Set config.reaction_checkpoint to a trained .pt file."
        )

    ckpt_path = Path(config.reaction_checkpoint)
    if not ckpt_path.exists():
        return f"[Reaction] Checkpoint not found: {ckpt_path}"

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state") or state.get("model_state_dict") or state)

    try:
        results = model.generate_reactions(reactant_data, num_samples=num_samples)
    except Exception as exc:
        return f"[Reaction] Generation failed: {exc}\n{traceback.format_exc()}"

    lines = [
        "Reaction Prediction",
        f"  Reactant: {reactant_smiles}  |  {len(results)} hypothesis(es)",
        "",
    ]
    for i, res in enumerate(results, 1):
        scores = res["reactive_scores"].squeeze(-1).cpu().tolist()
        top = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:5]
        lines.append(f"  Hypothesis {i}:")
        lines.append(f"    TS feature norm    : {float(res['ts_features'].norm()):.4f}")
        lines.append(f"    Product feature norm: {float(res['product_features'].norm()):.4f}")
        lines.append("    Top reactive atoms : " + ", ".join(f"atom {a} ({scores[a]:.3f})" for a in top))
        lines.append("")

    return "\n".join(lines)


def build_reaction_tool(config):
    """Return a LangChain Tool wrapping"""

    from langchain_core.tools import Tool

    def _run(query):
        query = query.strip()
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            params = {"smiles": query.split()[0] if query else ""}
        smiles = params.get("smiles", "").strip()
        if not smiles:
            return "[Reaction] 'smiles' is required."
        return predict_reaction(smiles, config, num_samples=params.get("num_samples"))

    return Tool(
        name="predict_reaction",
        func=_run,
        description=(
            "Predicts TS and product structures for a reactant using MPNN+CVAE. "
            "Input JSON: {\"smiles\": \"CCO\", \"num_samples\": 3}."
        ),
    )
