"""
Solvent-phase Molecular Geometry Prediction

Functions:
1. Load molecular features
3. MLP neural network model
4. Training function
5. Hyperparameter optimization
6. Geometry prediction

Usage: python py3_train.py --npz-path dataset_features.npz \
    --output-dir results \
    --epochs 10 \
    [--test-size 0.2] \
    [--val-size 0.1] \
    [--hidden-dims 128 256 128] \
    [--dropout 0.2] \
    [--batch-size 32] \
    [--lr 0.001] \
    [--patience 10] \
    [--optimize] \
    [--n-trials 10] \
    [--show-plot] \
    [--cuda] \
    [--seed 42]

Updates:
    02.11.2025 Initial script [Rangsiman Ketkaew]
"""

import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")


class MoleculeDataset(Dataset):
    """PyTorch Dataset for molecular geometries"""

    def __init__(self, features, targets, masks=None, transform=None):
        """
        Args:
            features: numpy array of shape (n_data, n_features)
            targets: numpy array of shape (n_data, n_targets)
            masks: numpy array of shape (n_data, max_atoms) - mask for valid atoms
            transform: optional transform to apply
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.masks = (
            torch.tensor(masks, dtype=torch.float32) if masks is not None else None
        )
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        if self.masks is not None:
            return x, y, self.masks[idx]
        else:
            return x, y


def prepare_dataset(
    npz_path,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    max_atoms=None,
    use_displacement=True,
):
    """
    Load dataset from CSV and split into train/val/test sets.
    Normalize data and create datsets with masks.

    Args:
        npz_path: Path to npz file containing feature vectors
        test_size: Fraction for test set
        val_size: Fraction for validation set (from training data)
        random_state: Random seed for reproducibility
        max_atoms: Maximum number of atoms to pad to (None = auto-detect)
        use_displacement: If True, predict displacement vectors instead of absolute coords

    Returns:
        Dictionary containing train, val, test datasets and metadata
    """

    npz = np.load(npz_path)
    X = npz["X"]
    y = npz["y"]
    masks = npz["masks"]

    # If max_atoms not provided, detect from masks
    if max_atoms is None:
        max_atoms = masks.shape[1]

    X_train, X_test, y_train, y_test, masks_train, masks_test, idx_train, idx_test = (
        train_test_split(
            X,
            y,
            masks,
            np.arange(len(X)),
            test_size=test_size,
            random_state=random_state,
        )
    )

    X_train, X_val, y_train, y_val, masks_train, masks_val, idx_train, idx_val = (
        train_test_split(
            X_train,
            y_train,
            masks_train,
            idx_train,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
        )
    )

    print(f"\nDataset splits:")
    print(f"- Training:   {len(X_train)} data points")
    print(f"- Validation: {len(X_val)} data points")
    print(f"- Test:       {len(X_test)} data points")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_set = MoleculeDataset(X_train, y_train, masks_train)
    val_set = MoleculeDataset(X_val, y_val, masks_val)
    test_set = MoleculeDataset(X_test, y_test, masks_test)

    metadata = {
        "n_features": X.shape[1],
        "n_targets": y.shape[1],
        "max_atoms": int(max_atoms),
        "use_displacement": use_displacement,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "train_indices": idx_train.tolist(),
        "val_indices": idx_val.tolist(),
        "test_indices": idx_test.tolist(),
    }

    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "metadata": metadata,
    }


class MoleculeMLP(nn.Module):
    """
    MLP for molecular geometry prediction

    Supports masking for variable-sized molecules
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[128, 256, 128],
        dropout=0.2,
        loss_fn=None,
        use_masking=True,
    ):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output values (flattened XYZ coordinates)
            hidden_dims: List of hidden layer sizes
            dropout: Dropout rate for regularization
            loss_fn: Loss function (default: MSELoss)
            use_masking: Whether to apply masking for variable-sized molecules
        """
        super(MoleculeMLP, self).__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(dims[-1], output_dim))

        self.model = nn.Sequential(*layers)
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss(reduction="none")
        self.use_masking = use_masking

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, pred, target, mask=None):
        """
        Compute loss with optional masking

        Args:
            pred: Predicted values (batch_size, output_dim)
            target: Target values (batch_size, output_dim)
            mask: Mask for valid atoms (batch_size, max_atoms)

        Returns:
            Scalar loss value
        """
        loss = self.loss_fn(pred, target)

        if self.use_masking and mask is not None:
            # Reshape mask to match coordinates (each atom has 3 coordinates)
            # mask: (batch_size, max_atoms) -> (batch_size, max_atoms * 3)
            mask_expanded = (
                mask.unsqueeze(-1).expand(-1, -1, 3).reshape(mask.size(0), -1)
            )

            loss = loss * mask_expanded

            return loss.sum() / (mask_expanded.sum() + 1e-8)
        else:
            return loss.mean()


def train(model, optimizer, train_loader, device="cpu"):
    """Train the model for one epoch

    Args:
        model: PyTorch model
        optimizer: optimizer
        data_loader: Training DataLoader
        device: device to use ("cpu" or "cuda")
    """
    model.train()
    train_loss = 0.0
    for batch_idx, batch_data in enumerate(train_loader):
        if len(batch_data) == 3:
            x, y, mask = batch_data
            x, y, mask = x.to(device), y.to(device), mask.to(device)
        else:
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            mask = None

        optimizer.zero_grad()
        pred = model(x)
        loss = model.compute_loss(pred, y, mask)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    device="cpu",
    save_path=None,
    patience=10,
):
    """
    Train the neural network model

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: "cpu" or "cuda"
        save_path: Path to save best model
        patience: Early stopping patience

    Returns:
        Dictionary containing training history
    """
    print(f"\nTraining model on {device}")
    print("---------------------")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "lr": []}

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train(model, optimizer, train_loader, device)
        used_time = time.time() - start_time
        val_loss = evaluate_model(model, val_loader, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:4d}/{epochs}: "
            f"train_loss = {train_loss:.4f} val_loss = {val_loss:.4f} "
            f"lr = {scheduler.get_last_lr()[0]:.4f} time {used_time:.2f}s"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    save_path,
                )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nTraining completed!")
    print(f"  Best validation loss: {best_val_loss:.6f}")

    return history


def evaluate_model(model, loader, device="cpu"):
    """Evaluate model on a dataset"""
    model.eval()  # to make sure that dropout/batchnorm are in eval mode
    total_loss = 0.0

    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                x, y, mask = batch_data
                x, y, mask = x.to(device), y.to(device), mask.to(device)
            else:
                x, y = batch_data
                x, y = x.to(device), y.to(device)
                mask = None

            pred = model(x)
            loss = model.compute_loss(pred, y, mask)
            total_loss += loss.item()

    return total_loss / len(loader)


def optimize_hyperparam(
    train_set, val_set, input_dim, output_dim, device="cpu", n_trials=10
):
    """
    Hyperparameter optimization using simple grid search

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        input_dim: Number of input features
        output_dim: Number of output values
        device: Device to train on ("cpu" or "cuda")
        n_trials: Number of trials per configuration

    Returns:
        Dictionary with best parameters and results
    """
    print("\nHyperparam optimization")
    print("-----------------------")

    param_grid = {
        "hidden_dims": [
            [64, 128, 64],
            [128, 256, 128],
            [256, 512, 256],
        ],
        "lr": [1e-2, 1e-3, 1e-4],
        "dropout": [0.1, 0.2, 0.3],
        "batch_size": [16, 32, 64],
    }

    best_val_loss = float("inf")
    best_params = None
    results = []

    trial = 0
    for hidden in param_grid["hidden_dims"]:
        for lr in param_grid["lr"]:
            for dropout in param_grid["dropout"]:
                for batch_size in param_grid["batch_size"]:
                    trial += 1
                    print(
                        f"\nTrial {trial}: hidden={hidden}, lr={lr}, "
                        f"dropout={dropout}, batch_size={batch_size}"
                    )

                    model = MoleculeMLP(
                        input_dim, output_dim, hidden_dims=hidden, dropout=dropout
                    )

                    train_loader = DataLoader(
                        train_set, batch_size=batch_size, shuffle=True
                    )
                    val_loader = DataLoader(val_set, batch_size=batch_size)

                    history = train_model(
                        model,
                        train_loader,
                        val_loader,
                        epochs=20,
                        lr=lr,
                        device=device,
                        patience=5,
                    )

                    val_loss = min(history["val_loss"])

                    results.append(
                        {
                            "hidden_dims": hidden,
                            "lr": lr,
                            "dropout": dropout,
                            "batch_size": batch_size,
                            "val_loss": val_loss,
                        }
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {
                            "hidden_dims": hidden,
                            "lr": lr,
                            "dropout": dropout,
                            "batch_size": batch_size,
                        }

                    if trial >= n_trials:
                        break
                if trial >= n_trials:
                    break
            if trial >= n_trials:
                break
        if trial >= n_trials:
            break

    print("Optimization results")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    return {
        "best_params": best_params,
        "best_val_loss": best_val_loss,
        "all_results": results,
    }


def predict_batch(model, test_loader, device="cpu"):
    """
    Make predictions on a batch of test data

    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device for prediction ("cpu" or "cuda")

    Returns:
        predictions, targets, masks
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                x, y, mask = batch_data
                x = x.to(device)
                all_masks.append(mask.numpy())
            else:
                x, y = batch_data
                x = x.to(device)
                mask = None

            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    masks = np.vstack(all_masks) if all_masks else None

    return predictions, targets, masks


def plot_history(history, save_path=None, show_plot=False):
    """Plot training and validation loss curves and also learning rate curve"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training History")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["lr"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_yscale("log")
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()


def analyze_prediction(prediction, target, masks=None, save_dir=None, show_plot=False):
    """
    Analyze the quality of prediction with various metrics and plots

    Args:
        prediction: Predicted values
        target: True values
        masks: Optional masks for valid atoms
        save_dir: Directory to save plots
        show_plot: Whether to display plots
    """
    print("\nPrediction analysis")
    print("-------------------")

    if masks is not None:
        masks_expanded = np.repeat(masks, 3, axis=1)
        # Calculate metrics only for valid (non-padded) atoms
        valid_preds = prediction[masks_expanded > 0]
        valid_targets = target[masks_expanded > 0]
        mse = np.mean((valid_preds - valid_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(valid_preds - valid_targets))
        errors = valid_preds - valid_targets
        all_err = prediction - target
    else:
        # Calculate metrics on all data
        mse = np.mean((prediction - target) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(prediction - target))
        errors = (prediction - target).flatten()
        all_err = prediction - target

    print(f"\nOverall Metrics:")
    print(f"  MSE:  {mse:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")

    n_data = len(prediction)
    n_coords = prediction.shape[1]
    max_atoms = n_coords // 3

    print(f"\nDataset size: {n_data} data")
    print(f"Max atoms per molecule: {max_atoms}")
    # if masks is not None:
    #     avg_atoms = masks.sum() / len(masks)
    #     print(f"Average atoms per molecule: {avg_atoms:.1f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ## 1. Predicted vs True scatter plot
    if masks is not None:
        plot_targets = valid_targets
        plot_preds = valid_preds
    else:
        plot_targets = target.flatten()
        plot_preds = prediction.flatten()

    axes[0, 0].scatter(plot_targets, plot_preds, alpha=0.3, s=1)
    axes[0, 0].plot(
        [plot_targets.min(), plot_targets.max()],
        [plot_targets.min(), plot_targets.max()],
        "r--",
        lw=2,
        label="Perfect prediction",
    )
    axes[0, 0].set_xlabel("True values")
    axes[0, 0].set_ylabel("Predicted values")
    axes[0, 0].set_title("Predicted vs True Coordinates")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    ## 2. Error distribution
    axes[0, 1].hist(errors, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Prediction Error")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Error Distribution")
    axes[0, 1].axvline(0, color="r", linestyle="--", lw=2)
    axes[0, 1].grid(True, alpha=0.3)

    ## 3. Error by coordinate (X, Y, Z)
    x_err = all_err[:, ::3].flatten()
    y_err = all_err[:, 1::3].flatten()
    z_err = all_err[:, 2::3].flatten()

    if masks is not None:
        mask_x = masks_expanded[:, ::3].flatten() > 0
        mask_y = masks_expanded[:, 1::3].flatten() > 0
        mask_z = masks_expanded[:, 2::3].flatten() > 0
        x_err = x_err[mask_x]
        y_err = y_err[mask_y]
        z_err = z_err[mask_z]

    axes[1, 0].boxplot([x_err, y_err, z_err], labels=["X", "Y", "Z"])
    axes[1, 0].set_ylabel("Error")
    axes[1, 0].set_title("Error by Coordinate Axis")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color="r", linestyle="--", lw=1)

    ## 4. RMSE per sample
    if masks is not None:
        rmse_per_sample = []
        for i in range(len(prediction)):
            mask_i = masks_expanded[i] > 0
            if mask_i.sum() > 0:
                rmse_i = np.sqrt(
                    np.mean((prediction[i][mask_i] - target[i][mask_i]) ** 2)
                )
                rmse_per_sample.append(rmse_i)
        rmse_per_sample = np.array(rmse_per_sample)
    else:
        rmse_per_sample = np.sqrt(np.mean((prediction - target) ** 2, axis=1))

    axes[1, 1].hist(rmse_per_sample, bins=30, edgecolor="black", alpha=0.7)
    axes[1, 1].set_xlabel("RMSE per molecule")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title(f"RMSE Distribution (Mean: {rmse_per_sample.mean():.4f})")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "prediction_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()


def main(args):
    """
    Main function to manage the training and prediction pipeline
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load feature set and prepare datasets

    data = prepare_dataset(
        args.npz_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )

    train_set = data["train"]
    val_set = data["val"]
    test_set = data["test"]
    metadata = data["metadata"]

    # Step 2: Hyperparameter optimization (optional)
    # otherwise we use default parameters

    if args.optimize:
        opt_results = optimize_hyperparam(
            train_set,
            val_set,
            metadata["n_features"],
            metadata["n_targets"],
            device=device,
            n_trials=args.n_trials,
        )

        best_params = opt_results["best_params"]

        opt_path = output_dir / "optimization_results.json"
        with open(opt_path, "w") as f:
            json.dump(
                {k: v for k, v in opt_results.items() if k != "all_results"},
                f,
                indent=2,
            )
    else:
        best_params = {
            "hidden_dims": args.hidden_dims,
            "lr": args.lr,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
        }
        print("\nUsing default parameters (skipping optimization)")

    print(f"\nModel parameters:")
    for key, value in best_params.items():
        print(f"- {key}: {value}")

    metadata["best_hidden_dims"] = best_params["hidden_dims"]
    metadata["best_dropout"] = best_params["dropout"]

    # Step 3: Create and train model

    model = MoleculeMLP(
        input_dim=metadata["n_features"],
        output_dim=metadata["n_targets"],
        hidden_dims=best_params["hidden_dims"],
        dropout=best_params["dropout"],
    )

    # print(f"\nModel architecture:")
    # print(model)
    print(f"Best hidden_dims  : {best_params['hidden_dims']}")
    print(f"Best dropout      : {best_params['dropout']}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(
        train_set, batch_size=best_params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=best_params["batch_size"])
    test_loader = DataLoader(test_set, batch_size=best_params["batch_size"])

    model_path = output_dir / "best_model.pt"
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=best_params["lr"],
        device=device,
        save_path=model_path,
        patience=args.patience,
    )

    plot_history(
        history, save_path=output_dir / "training_history.png", show_plot=args.show_plot
    )

    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Step 4: Load the best model and evaluate

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best model from epoch {checkpoint["epoch"]+1}")

    test_loss = evaluate_model(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.6f}")

    # Step 5: Make predictions on test set

    predictions, targets, masks = predict_batch(model, test_loader, device)

    analyze_prediction(
        predictions, targets, masks, save_dir=output_dir, show_plot=args.show_plot
    )

    # Step 6: Save metadata and final results

    print(f"\nOutput directory: {output_dir}")

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    results = {
        "test_loss": test_loss,
        "best_params": best_params,
        "model_path": str(model_path),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Predicting molecular geometry in solvent using a neural network.")
    parser.add_argument("--npz-path", type=str, required=True, help="Path to dataset NPZ file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for test set")
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction of training data for validation")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 256, 128], help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--optimize", action="store_true", help="Perform hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials for optimization")
    parser.add_argument("--show-plot", action="store_true", help="Show plots interactively")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for results")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    args = parser.parse_args()
    # fmt: on

    main(args)
