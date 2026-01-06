"""
Visualize predicted solvent phase geometries against true geometries.

Usage: python py4_visualize_prediction.py \
    --model-path results/best_model.pt \
    --csv-path molecule_dataset.csv \
    --metadata-path results/metadata.json \
    [--n-visualize 3] \
    [--output-dir visualizations] \
    [--show-plot]

Updates:
    03.11.2025 Initial script [Rangsiman Ketkaew]
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from matplotlib.patches import Patch
from pathlib import Path
from py2_create_feature_set import calculate_mol_features
from py3_train import MoleculeMLP

# fmt: off
atom_symbols = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn"
}
# fmt: on


def predict_geometry(
    model,
    gas_xyz,
    atomic_numbers,
    dielectric,
    scaler_mean,
    scaler_scale,
    device="cpu",
    use_displacement=True,
):
    """
    Predict solvent phase geometry given gas phase geometry

    Args:
        model: Trained PyTorch model
        gas_xyz: Gas phase coordinates ((n_atoms, 3) numpy array)
        atomic_numbers: Atomic numbers (1D numpy array)
        dielectric: Dielectric constant of solvent (float)
        scaler_mean: Mean for feature normalization
        scaler_scale: Scale for feature normalization
        device: "cpu" or "cuda"
        use_displacement: If True, model predicts displacement vectors

    Returns:
        numpy array of shape (max_atoms, 3) - predicted solvent phase coordinates

        Note: May include padding. Use only first n_atoms rows for actual molecule.
    """
    model.eval()

    features = calculate_mol_features(gas_xyz, atomic_numbers, dielectric)
    # lazy normalize features using scaler_mean and scaler_scale
    features = (features.reshape(1, -1) - scaler_mean) / scaler_scale

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).to(device)
        pred = model(x)
        pred_flat = pred.cpu().numpy().squeeze()  # (max_atoms * 3,)

    pred_xyz = pred_flat.reshape(-1, 3)

    if use_displacement:
        n_atoms = len(gas_xyz)
        max_atoms = len(pred_xyz)
        gas_xyz_padded = np.zeros((max_atoms, 3))
        gas_xyz_padded[:n_atoms] = gas_xyz
        pred_xyz = gas_xyz_padded + pred_xyz

    return pred_xyz


def save_xyz(filepath, atomic_nums, coords, comment=""):
    """
    Save coordinates in .xyz file format

    Args:
        filepath: Path to save the XYZ file
        atomic_nums: Atomic numbers (1D numpy array)
        coords: Coordinates ((n_atoms, 3) numpy array)
        comment: Comment line (optional)
    """
    with open(filepath, "w") as f:
        f.write(f"{len(atomic_nums)}\n")
        f.write(f"{comment}\n")
        for i, (coords, z) in enumerate(zip(coords, atomic_nums)):
            symbol = atom_symbols.get(z, f"Z{z}")
            f.write(
                f"{symbol:3s} {coords[0]:12.6f} {coords[1]:12.6f} {coords[2]:12.6f}\n"
            )


def visualize_molecule(
    gas_xyz, pred_xyz, true_xyz, atomic_numbers, save_path=None, show_plot=False
):
    """
    Visualize the prediction of a single molecule

    Args:
        gas_xyz: Gas phase coordinates
        pred_xyz: Predicted solvent phase coordinates
        true_xyz: True solvent phase coordinates
        atomic_numbers: Atomic numbers
        save_path: Path to save figure
        show_plot: Whether to display the plot
    """
    fig = plt.figure(figsize=(15, 6))

    colors = {1: "white", 6: "gray", 7: "blue", 8: "red"}
    atom_colors = [colors.get(z, "green") for z in atomic_numbers]

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(*gas_xyz.T, c=atom_colors, s=100, alpha=0.8, edgecolors="black")
    ax1.set_title("Gas Phase")
    ax1.set_xlabel("X (Å)")
    ax1.set_ylabel("Y (Å)")
    ax1.set_zlabel("Z (Å)")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(*pred_xyz.T, c=atom_colors, s=100, alpha=0.8, edgecolors="black")
    ax2.set_title("Predicted Solvent Phase")
    ax2.set_xlabel("X (Å)")
    ax2.set_ylabel("Y (Å)")
    ax2.set_zlabel("Z (Å)")

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.scatter(*true_xyz.T, c=atom_colors, s=100, alpha=0.8, edgecolors="black")
    ax3.set_title("True Solvent Phase")
    ax3.set_xlabel("X (Å)")
    ax3.set_ylabel("Y (Å)")
    ax3.set_zlabel("Z (Å)")

    legend_elements = []
    unique_atoms = sorted(set(atomic_numbers))
    for z in unique_atoms:
        color = colors.get(z, "green")
        label = atom_symbols.get(z, f"Z={z}")
        legend_elements.append(Patch(facecolor=color, edgecolor="black", label=label))

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        # bbox_to_anchor=(0.5, -0.05),
        frameon=True,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Create NumPy feature set from dataset CSV")
    parser.add_argument("--model-path", required=True, help="Path to trained model file")
    parser.add_argument("--csv-path", required=True, help="Input dataset CSV file")
    parser.add_argument("--metadata-path", required=True, help="Metadata JSON file")
    parser.add_argument("--n-visualize", type=int, default=3, help="Number of molecules to visualize")
    parser.add_argument("--output-dir", default="visualizations", help="Output directory for visualizations")
    parser.add_argument("--save-xyz", action="store_true", help="Whether to save XYZ files")
    parser.add_argument("--show-plot", action="store_true", help="Whether to show the plot interactively")
    args = parser.parse_args()
    # fmt: on

    print("Visualizing predictions")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = json.load(open(args.metadata_path, "r"))
    scaler_mean = metadata["scaler_mean"]
    scaler_scale = metadata["scaler_scale"]
    best_hidden_dims = metadata["best_hidden_dims"]
    best_dropout = metadata["best_dropout"]

    model = MoleculeMLP(
        input_dim=metadata["n_features"],
        output_dim=metadata["n_targets"],
        hidden_dims=metadata["best_hidden_dims"],
        dropout=metadata["best_dropout"],
    )

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    df = pd.read_csv(args.csv_path)
    df["atomic_numbers"] = df["atomic_numbers"].apply(eval)
    df["gas_xyz"] = df["gas_xyz"].apply(eval)
    df["sol_xyz"] = df["sol_xyz"].apply(eval)

    test_indices = metadata["test_indices"][: args.n_visualize]

    for i, idx in enumerate(test_indices):
        print(f"- Molecule {i+1}/{len(test_indices)} (index {idx})")

        row = df.iloc[idx]
        gas_xyz = np.array(row["gas_xyz"]).reshape(-1, 3)
        true_xyz = np.array(row["sol_xyz"]).reshape(-1, 3)
        atomic_nums = np.array(row["atomic_numbers"])
        dielectric = row["dielectric_const"]

        pred_xyz = predict_geometry(
            model,
            gas_xyz,
            atomic_nums,
            dielectric,
            scaler_mean,
            scaler_scale,
            device,
            use_displacement=metadata.get("use_displacement", True),
        )

        n_atoms = len(atomic_nums)
        pred_xyz = pred_xyz[:n_atoms]

        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        if args.save_xyz:
            gas_xyz_path = output_dir / f"struc_molecule_{idx}_gas.xyz"
            pred_xyz_path = output_dir / f"struc_molecule_{idx}_predicted_solvent.xyz"
            true_xyz_path = output_dir / f"struc_molecule_{idx}_true_solvent.xyz"

            save_xyz(gas_xyz_path, atomic_nums, gas_xyz, "Gas phase geometry")
            save_xyz(
                pred_xyz_path, atomic_nums, pred_xyz, "Predicted solvent phase geometry"
            )
            save_xyz(
                true_xyz_path, atomic_nums, true_xyz, "True solvent phase geometry"
            )

        vis_path = output_dir / f"molecule_{idx}_prediction.png"
        visualize_molecule(
            gas_xyz,
            pred_xyz,
            true_xyz,
            atomic_nums,
            save_path=vis_path,
            show_plot=args.show_plot,
        )
