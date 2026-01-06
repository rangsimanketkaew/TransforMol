"""
This script is to create features dataset (in numpy npz) for molecular geometry dataset from CSV dataset file

After you generate the dataset using `generate_dataset.py`, you can use this script to create the feature set.

Usage: python py2_create_feature_set.py \
    --csv-path molecule_dataset.csv \
    --output-npz dataset_features.npz \
    [--max-atoms 50] \
    [--use-absolute-coord]

Updates:
    03.11.2025 Initial script [Rangsiman Ketkaew]
"""

import argparse
import numpy as np
import pandas as pd


def calculate_mol_features(xyz, atomic_numbers, dielectric):
    """
    Calculate global molecular descriptors that capture structural properties
    These features are independent of the number of atoms

    Args:
        xyz: Cartesian coordinates (numpy array)
        atomic_numbers: Atomic numbers (numpy array)
        dielectric: Dielectric constant (float)

    Returns:
        Feature vector (numpy array)
    """
    n_atoms = len(xyz)

    centroid = xyz.mean(axis=0)
    distances = np.linalg.norm(xyz - centroid, axis=1)
    max_dist = distances.max()
    min_dist = distances.min()
    mean_dist = distances.mean()
    std_dist = distances.std()

    radius_gyration = np.sqrt(np.mean(distances**2))

    pairwise_dists = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pairwise_dists.append(np.linalg.norm(xyz[i] - xyz[j]))

    pairwise_dists = np.array(pairwise_dists)
    max_pairwise = pairwise_dists.max() if len(pairwise_dists) > 0 else 0.0
    min_pairwise = pairwise_dists.min() if len(pairwise_dists) > 0 else 0.0
    mean_pairwise = pairwise_dists.mean() if len(pairwise_dists) > 0 else 0.0
    std_pairwise = pairwise_dists.std() if len(pairwise_dists) > 0 else 0.0

    I = np.zeros((3, 3))
    for coord in xyz - centroid:
        I += np.outer(coord, coord)
    eigenvalues = np.linalg.eigvalsh(I)

    I_1, I_2, I_3 = sorted(eigenvalues)
    asphericity = I_3 - 0.5 * (I_1 + I_2) if I_3 > 0 else 0.0

    # unique_atoms, counts = np.unique(atomic_numbers, return_counts=True)
    n_heavy_atoms = np.sum(atomic_numbers > 1)
    n_hydrogen = np.sum(atomic_numbers == 1)

    masses = atomic_numbers
    com = np.average(xyz, weights=masses, axis=0)
    com_distances = np.linalg.norm(xyz - com, axis=1)

    features = [
        n_atoms,  # Number of atoms
        n_heavy_atoms,  # Number of heavy (non-hydrogen) atoms
        n_hydrogen,  # Number of hydrogen atoms
        dielectric,  # Dielectric constant
        max_dist,  # Max distance from centroid
        min_dist,  # Min distance from centroid
        mean_dist,  # Mean distance from centroid
        std_dist,  # Std distance from centroid
        radius_gyration,  # Radius of gyration
        max_pairwise,  # Max pairwise distance
        min_pairwise,  # Min pairwise distance
        mean_pairwise,  # Mean pairwise distance
        std_pairwise,  # Std pairwise distance
        I_1,  # Moments of inertia
        I_2,
        I_3,
        asphericity,  # Asphericity
        centroid[0],  # Centroid position
        centroid[1],
        centroid[2],
        com[0],  # Center of mass (COM) position
        com[1],
        com[2],
        np.mean(com_distances),  # Mean distance from COM
        np.std(com_distances),  # Std distance from COM
    ]

    return np.array(features, dtype=np.float32)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Create NumPy feature set from dataset CSV")
    parser.add_argument("--csv-path", required=True, help="Path to dataset CSV file")
    parser.add_argument("--max-atoms", type=int, default=None, help="Maximum number of atoms (for padding)")
    parser.add_argument("--use-absolute-coord", action="store_true", help="Use displacement vectors as target")
    parser.add_argument("--output-npz", type=str,default="dataset_features.npz", help="Output NPZ file for features")
    args = parser.parse_args()
    # fmt: on

    df = pd.read_csv(args.csv_path)

    print(f"Total data: {len(df)}")

    # Convert string representations to lists
    df["atomic_numbers"] = df["atomic_numbers"].apply(eval)
    df["gas_xyz"] = df["gas_xyz"].apply(eval)
    df["sol_xyz"] = df["sol_xyz"].apply(eval)

    if args.max_atoms is None:
        args.max_atoms = df["num_atoms"].max()
        print(f"Auto-detected max_atoms: {args.max_atoms}")

    print(f"Using max_atoms: {args.max_atoms}")
    print(
        f"Prediction mode: {"displacement vectors" if not args.use_absolute_coord else "absolute coordinates"}"
    )

    features_list = []
    targets_list = []
    masks_list = []

    # looping through each molecule in the dataset via df.iterrows() is not recommended
    # because it can be slow for large datasets. However, we use it here for simplicity.
    # We could use vectorization for better performance.
    for idx, row in df.iterrows():
        print(
            f"Calculating features of molecule {idx + 1}/{len(df)}",
            end="\r",
            flush=True,
        )
        gas_xyz = np.array(row["gas_xyz"]).reshape(-1, 3)
        sol_xyz = np.array(row["sol_xyz"]).reshape(-1, 3)
        atomic_nums = np.array(row["atomic_numbers"])
        dielectric = row["dielectric_const"]
        n_atoms = len(atomic_nums)

        features = calculate_mol_features(gas_xyz, atomic_nums, dielectric)

        if not args.use_absolute_coord:
            # Target: displacement vectors (the devitation from gas to solvent)
            displacement = sol_xyz - gas_xyz
            target = displacement.flatten()
        else:
            # Target: absolute solvent XYZ coordinates
            target = sol_xyz.flatten()

        # Pad target to max_atoms * 3
        padded_target = np.zeros(args.max_atoms * 3, dtype=np.float32)
        padded_target[: len(target)] = target

        # Create mask for actual atoms (1 for real atoms, 0 for padding)
        mask = np.zeros(args.max_atoms, dtype=np.float32)
        mask[:n_atoms] = 1.0

        features_list.append(features)
        targets_list.append(padded_target)
        masks_list.append(mask)

    print("\nMolecular features:")
    X = np.array(features_list)
    print(f"- Feature shape: {X.shape}")
    y = np.array(targets_list)
    print(f"- Target shape : {y.shape}")
    masks = np.array(masks_list)
    print(f"- Mask shape   : {masks.shape}")

    np.savez_compressed(args.output_npz, X=X, y=y, masks=masks)
