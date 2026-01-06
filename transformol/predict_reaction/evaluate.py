"""
Evaluation functions for reaction prediction.

Updates:
    14.10.2025 Initial script [Rangsiman Ketkaew]
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data_loader import load_datasets, ReactionDataset
from model import ReactionGenerativeModel
from train import collate_fn, identify_reactive_atoms, extract_single_graph


def rmsd(coords_1, coords_2):
    if coords_1.shape != coords_2.shape:
        return float("inf")

    diff = coords_1 - coords_2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def mae(features_1, features_2):
    return torch.mean(torch.abs(features_1 - features_2)).item()


def evaluate(model, test_loader, device):
    model.eval()

    ts_mae_list = []
    product_mae_list = []
    reactive_acc = []
    reactive_prec = []
    reactive_recall = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            r_data = batch["reactant"].to(device)
            ts_data = batch["ts"].to(device)
            p_data = batch["product"].to(device)

            reactive_labels_list = []
            ptr = batch["reactant"].ptr
            batch_size = len(ptr) - 1

            for i in range(batch_size):
                start = ptr[i].item()
                end = ptr[i + 1].item()

                r_single = extract_single_graph(r_data, start, end)
                ts_single = extract_single_graph(ts_data, start, end)
                p_single = extract_single_graph(p_data, start, end)

                reactive_labels = identify_reactive_atoms(r_single, ts_single, p_single)
                reactive_labels_list.append(reactive_labels)

            reactive_labels = torch.cat(reactive_labels_list).to(device)

            output = model(r_data, ts_data, p_data, training=True)
            ts_mae = mae(output["ts_node_features"], ts_data.x)
            ts_mae_list.append(ts_mae)

            p_mae = mae(output["p_node_features"], p_data.x)
            product_mae_list.append(p_mae)

            pred_reactive = (output["reactive_scores"] > 0.5).float().squeeze(-1)

            accuracy = (pred_reactive == reactive_labels).float().mean().item()
            reactive_acc.append(accuracy)

            true_pos = ((pred_reactive == 1) & (reactive_labels == 1)).float().sum()
            pred_pos = (pred_reactive == 1).float().sum()
            precision = (true_pos / pred_pos).item() if pred_pos > 0 else 0.0
            reactive_prec.append(precision)

            actual_pos = (reactive_labels == 1).float().sum()
            recall = (true_pos / actual_pos).item() if actual_pos > 0 else 0.0
            reactive_recall.append(recall)

            rxn_ids = batch.get("rxn_ids", batch.get("rxn_id", []))
            if not isinstance(rxn_ids, list):
                rxn_ids = [rxn_ids]
            all_predictions.append(
                {
                    "rxn_ids": rxn_ids,
                    "reactive_scores": output["reactive_scores"].cpu().numpy(),
                    "reactive_labels": reactive_labels.cpu().numpy(),
                }
            )

    metrics = {
        "ts_mae": np.mean(ts_mae_list),
        "ts_mae_std": np.std(ts_mae_list),
        "product_mae": np.mean(product_mae_list),
        "product_mae_std": np.std(product_mae_list),
        "reactive_acc": np.mean(reactive_acc),
        "reactive_prec": np.mean(reactive_prec),
        "reactive_recall": np.mean(reactive_recall),
    }

    if metrics["reactive_prec"] + metrics["reactive_recall"] > 0:
        metrics["reactive_f1"] = (
            2
            * metrics["reactive_prec"]
            * metrics["reactive_recall"]
            / (metrics["reactive_prec"] + metrics["reactive_recall"])
        )
    else:
        metrics["reactive_f1"] = 0.0

    return metrics, all_predictions


def generate_reactions(
    model, reactant_data, rxn_id, num_samples=3, save_dir="./predictions"
):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    reactant_data = reactant_data.to(device)

    generated_reactions = model.generate_reactions(
        reactant_data, num_samples=num_samples
    )

    results = {
        "rxn_id": rxn_id,
        "num_atoms": reactant_data.x.shape[0],
        "reactive_scores": generated_reactions[0]["reactive_scores"]
        .cpu()
        .numpy()
        .tolist(),
        "samples": [],
    }

    for i, reaction in enumerate(generated_reactions):
        sample_result = {
            "sample_id": i,
            "ts_features": reaction["ts_features"].cpu().numpy().tolist(),
            "product_features": reaction["product_features"].cpu().numpy().tolist(),
        }
        results["samples"].append(sample_result)

    output_path = os.path.join(save_dir, f"reaction_{rxn_id}_predictions.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return generated_reactions


def visualize_reactive_atoms(
    reactant_data, reactive_scores, rxn_id, save_dir="./visualizations"
):
    os.makedirs(save_dir, exist_ok=True)

    atom_indices = np.arange(len(reactive_scores))
    colors = ["red" if score > 0.5 else "blue" for score in reactive_scores]

    plt.figure(figsize=(10, 5))
    plt.bar(atom_indices, reactive_scores, color=colors, alpha=0.7)
    plt.axhline(y=0.5, color="k", linestyle="--", label="Threshold")
    plt.xlabel("Atom Index")
    plt.ylabel("Reactive Score")
    plt.title(f"Reactive Atom Predictions for Reaction {rxn_id}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"reactive_atoms_{rxn_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_eval_results(metrics, save_path="./evaluation_results.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics_names = ["TS MAE", "Product MAE"]
    values = [metrics["ts_mae"], metrics["product_mae"]]
    errors = [metrics["ts_mae_std"], metrics["product_mae_std"]]

    axes[0].bar(
        metrics_names,
        values,
        yerr=errors,
        capsize=5,
        alpha=0.7,
        color=["blue", "green"],
    )
    axes[0].set_ylabel("Mean Absolute Error")
    axes[0].set_title("Feature Reconstruction Performance")
    axes[0].grid(True, alpha=0.3)

    reactive_metrics = ["Accuracy", "Precision", "Recall", "F1"]
    reactive_values = [
        metrics["reactive_acc"],
        metrics["reactive_prec"],
        metrics["reactive_recall"],
        metrics["reactive_f1"],
    ]

    axes[1].bar(
        reactive_metrics,
        reactive_values,
        alpha=0.7,
        color=["purple", "orange", "red", "brown"],
    )
    axes[1].set_ylabel("Score")
    axes[1].set_title("Reactive Atom Prediction Performance")
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


def test_on_examples(
    model,
    test_set,
    num_examples=5,
    num_reaction_predictions=3,
    save_dir="./examples",
):
    os.makedirs(save_dir, exist_ok=True)

    device = next(model.parameters()).device

    print("")
    print(
        f"Prediction on {num_examples} reactants and generating {num_reaction_predictions} plausible reactions..."
    )

    for i in range(min(num_examples, len(test_set))):
        sample = test_set.get(i)
        rxn_id = sample["rxn_id"]

        print(f"{"="*50}")
        print(f"{i+1}) Reaction no. {rxn_id}:")
        print(f"  Number of atoms: {sample["reactant"].x.shape[0]}")

        generated = generate_reactions(
            model,
            sample["reactant"],
            rxn_id,
            num_samples=num_reaction_predictions,
            save_dir=save_dir,
        )

        reactive_scores = generated[0]["reactive_scores"].cpu().numpy().squeeze()

        print(
            f"  Predicted reactive atoms: {np.where(reactive_scores > 0.5)[0].tolist()}"
        )

        visualize_reactive_atoms(
            sample["reactant"], reactive_scores, rxn_id, save_dir=save_dir
        )


if __name__ == "__main__":
    ###### User defined Parameters ######
    dataset_dir = "./datasets"
    checkpoint = "./checkpoints/best_model.pt"
    results_dir = "./results"
    num_examples = 5
    num_reaction_predictions = 3
    #####################################

    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_dataset = load_datasets(dataset_dir)

    print(f"Test dataset size: {len(test_dataset)}")

    sample = test_dataset.get(0)
    node_dim = sample["reactant"].x.shape[1]
    edge_dim = sample["reactant"].edge_attr.shape[1]

    model = ReactionGenerativeModel(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        hidden_dim=128,
        latent_dim=64,
        num_mpnn_layers=4,
    ).to(device)

    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("")
    print(f"Model loaded from epoch {checkpoint["epoch"] + 1}")
    print(f"  Training loss: {checkpoint["train_loss"]:.4f}")
    print(f"  Validation loss: {checkpoint["val_loss"]:.4f}")

    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    metrics, predictions = evaluate(model, test_loader, device)

    print("\n-------------------")
    print("Evaluation Results:")
    print("-------------------\n")
    print(f"TS feature MAE: {metrics["ts_mae"]:.4f} ± {metrics["ts_mae_std"]:.4f}")
    print(
        f"Product feature MAE: {metrics["product_mae"]:.4f} ± {metrics["product_mae_std"]:.4f}"
    )
    print(f"\nReactive Atom Prediction:")
    print(f"  Accuracy:  {metrics["reactive_acc"]:.4f}")
    print(f"  Precision: {metrics["reactive_prec"]:.4f}")
    print(f"  Recall:    {metrics["reactive_recall"]:.4f}")
    print(f"  F1 Score:  {metrics["reactive_f1"]:.4f}")

    metrics_path = os.path.join(results_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_eval_results(
        metrics, save_path=os.path.join(results_dir, "evaluation_results.png")
    )

    test_on_examples(
        model,
        test_dataset,
        num_examples=num_examples,
        num_reaction_predictions=num_reaction_predictions,
        save_dir=os.path.join(results_dir, "examples"),
    )
