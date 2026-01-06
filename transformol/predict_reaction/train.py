"""
Training script for the reaction generative model
"""

import os
import json
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data_loader import create_datasets, load_datasets, ReactionDataset
from model import ReactionGenerativeModel, calculate_loss


def identify_reactive_atoms(reactant_data, ts_data, product_data):
    """
    Identify reactive atoms by comparing reactant and product connectivity.

    A simple heuristic: atoms with different bonding patterns between reactant and product are considered reactive.

    Args:
        reactant_data: Reactant graph
        ts_data: TS graph
        product_data: Product graph

    Returns:
        reactive_labels: Binary tensor (N,) indicating reactive atoms
    """
    num_atoms = reactant_data.x.size(0)

    # Get adjacency information
    r_edges = set(
        tuple(sorted([i.item(), j.item()])) for i, j in reactant_data.edge_index.t()
    )
    p_edges = set(
        tuple(sorted([i.item(), j.item()])) for i, j in product_data.edge_index.t()
    )

    # Bonds that change (break or form)
    changed_bonds = r_edges.symmetric_difference(p_edges)

    # Atoms involved in changed bonds are reactive
    reactive_atoms = set()
    for i, j in changed_bonds:
        reactive_atoms.add(i)
        reactive_atoms.add(j)

    # Create binary labels
    reactive_labels = torch.zeros(num_atoms, dtype=torch.float32)
    for atom_idx in reactive_atoms:
        reactive_labels[atom_idx] = 1.0

    return reactive_labels


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Args:
        batch: List of data dictionaries

    Returns:
        Batched data
    """
    if isinstance(batch[0], dict):
        reactants = [item["reactant"] for item in batch]
        ts_list = [item["ts"] for item in batch]
        products = [item["product"] for item in batch]
        rxn_ids = [item["rxn_id"] for item in batch]
    else:
        batch_dicts = batch
        reactants = [item["reactant"] for item in batch_dicts]
        ts_list = [item["ts"] for item in batch_dicts]
        products = [item["product"] for item in batch_dicts]
        rxn_ids = [item["rxn_id"] for item in batch_dicts]

    from torch_geometric.data import Batch

    r_batch = Batch.from_data_list(reactants)
    ts_batch = Batch.from_data_list(ts_list)
    p_batch = Batch.from_data_list(products)

    return {"reactant": r_batch, "ts": ts_batch, "product": p_batch, "rxn_ids": rxn_ids}


def train(model, train_loader, optimizer, device, beta=0.001):
    """
    Train for one epoch

    Args:
        model: Reaction generative model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        beta: Weight for KL divergence

    Returns:
        Dictionary with average losses
    """
    model.train()

    total_loss = 0
    ts_recon_loss = 0
    p_recon_loss = 0
    ts_kl_loss = 0
    p_kl_loss = 0
    reactive_loss = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training"):
        r_data = batch["reactant"].to(device)
        ts_data = batch["ts"].to(device)
        p_data = batch["product"].to(device)

        # Identify reactive atoms
        reactive_labels_list = []
        ptr = batch["reactant"].ptr  # Batch pointer
        batch_size = len(ptr) - 1  # ptr has batch_size + 1 elements

        for i in range(batch_size):
            start = ptr[i].item()
            end = ptr[i + 1].item()

            # Extract individual graphs
            r_single = extract_single_graph(r_data, start, end)
            ts_single = extract_single_graph(ts_data, start, end)
            p_single = extract_single_graph(p_data, start, end)

            reactive_labels = identify_reactive_atoms(r_single, ts_single, p_single)
            reactive_labels_list.append(reactive_labels)

        reactive_labels = torch.cat(reactive_labels_list).to(device)

        output = model(r_data, ts_data, p_data, training=True)

        targets = {
            "ts_features": ts_data.x,
            "p_features": p_data.x,
            "reactive_labels": reactive_labels,
        }

        losses = calculate_loss(output, targets, beta=beta)
        optimizer.zero_grad()
        losses["total_loss"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += losses["total_loss"].item()
        ts_recon_loss += losses["ts_recon_loss"].item()
        p_recon_loss += losses["p_recon_loss"].item()
        ts_kl_loss += losses["ts_kl_loss"].item()
        p_kl_loss += losses["p_kl_loss"].item()
        reactive_loss += losses["reactive_loss"].item()
        num_batches += 1

    return {
        "total_loss": total_loss / num_batches,
        "ts_recon_loss": ts_recon_loss / num_batches,
        "p_recon_loss": p_recon_loss / num_batches,
        "ts_kl_loss": ts_kl_loss / num_batches,
        "p_kl_loss": p_kl_loss / num_batches,
        "reactive_loss": reactive_loss / num_batches,
    }


def extract_single_graph(batch_data, start_idx, end_idx):
    """Extract a single graph from a batched graph"""
    from torch_geometric.data import Data

    # Get node features
    x = batch_data.x[start_idx:end_idx]

    # Get edges within this graph
    edge_mask = (batch_data.edge_index[0] >= start_idx) & (
        batch_data.edge_index[0] < end_idx
    )
    edge_index = batch_data.edge_index[:, edge_mask] - start_idx
    edge_attr = batch_data.edge_attr[edge_mask]

    # Get positions if available
    # if hasattr(batch_data, "pos"):
    #     pos = batch_data.pos[start_idx:end_idx]
    #     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    # else:
    #     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def predict(model, val_loader, device, beta=0.001):
    model.eval()

    total_loss = 0
    ts_recon_loss = 0
    p_recon_loss = 0
    reactive_acc = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            r_data = batch["reactant"].to(device)
            ts_data = batch["ts"].to(device)
            p_data = batch["product"].to(device)

            # Identify reactive atoms
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

            targets = {
                "ts_features": ts_data.x,
                "p_features": p_data.x,
                "reactive_labels": reactive_labels,
            }

            losses = calculate_loss(output, targets, beta=beta)

            # Reactive atom accuracy
            pred_reactive = (output["reactive_scores"] > 0.5).float().squeeze(-1)
            correct = (pred_reactive == reactive_labels).float().mean()

            total_loss += losses["total_loss"].item()
            ts_recon_loss += losses["ts_recon_loss"].item()
            p_recon_loss += losses["p_recon_loss"].item()
            reactive_acc += correct.item()
            num_batches += 1

    return {
        "total_loss": total_loss / num_batches,
        "ts_recon_loss": ts_recon_loss / num_batches,
        "p_recon_loss": p_recon_loss / num_batches,
        "reactive_acc": reactive_acc / num_batches,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=5,
    lr=1e-3,
    beta=0.001,
    save_dir="./checkpoints",
    device="cuda",
):
    """
    Args:
        model: Reaction generative model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr: Learning rate
        beta: Weight for KL divergence
        save_dir: Directory to save checkpoints
        device: Device to train on ("cpu" or "cuda")
    """
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    history = {"train_loss": [], "val_loss": [], "val_reactive_acc": []}
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_metrics = train(model, train_loader, optimizer, device, beta)
        val_metrics = predict(model, val_loader, device, beta)
        scheduler.step(val_metrics["total_loss"])

        # fmt: off
        print(f"Train loss: {train_metrics["total_loss"]:.4f}")
        print(f"  TS Recon: {train_metrics["ts_recon_loss"]:.4f}, P Recon: {train_metrics["p_recon_loss"]:.4f}")
        print(f"  TS KL: {train_metrics["ts_kl_loss"]:.4f}, P KL: {train_metrics["p_kl_loss"]:.4f}")
        print(f"  Reactive: {train_metrics["reactive_loss"]:.4f}")
        print(f"Val loss: {val_metrics["total_loss"]:.4f}")
        print(f"  TS Recon: {val_metrics["ts_recon_loss"]:.4f}, P Recon: {val_metrics["p_recon_loss"]:.4f}")
        print(f"  Reactive Acc: {val_metrics["reactive_acc"]:.4f}")
        # fmt: on

        history["train_loss"].append(train_metrics["total_loss"])
        history["val_loss"].append(val_metrics["total_loss"])
        history["val_reactive_acc"].append(val_metrics["reactive_acc"])
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics["total_loss"],
            "val_loss": val_metrics["total_loss"],
            "history": history,
        }

        # torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
            print(f"  New best model saved! (Val Loss: {best_val_loss:.4f})")

    torch.save(checkpoint, os.path.join(save_dir, "final_model.pt"))

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


def plot_training_history(history, save_path="./training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history["val_reactive_acc"],
        label="Reactive Atom Acc",
        marker="o",
        color="green",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Reactive Atom Prediction Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


if __name__ == "__main__":
    ###### User defined Parameters ######
    dataset_dir = "./datasets"
    checkpoint_dir = "./checkpoints"
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-3
    beta = 0.001  # KL divergence weight
    #####################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, val_dataset, _ = load_datasets(dataset_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Get feature dimensions from first sample
    sample = train_dataset.get(0)
    node_dim = sample["reactant"].x.shape[1]
    edge_dim = sample["reactant"].edge_attr.shape[1]

    print(f"\nFeature dimensions:")
    print(f" Node features: {node_dim}")
    print(f" Edge features: {edge_dim}")

    model = ReactionGenerativeModel(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        hidden_dim=128,
        latent_dim=64,
        num_mpnn_layers=4,
    ).to(device)

    print(
        f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        beta=beta,
        save_dir=checkpoint_dir,
        device=device,
    )

    plot_training_history(history, save_path="./training_history.png")

    print("")
    print("Training done!")
