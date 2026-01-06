"""
Data loading and preprocessing for reactive atom prediction.

Updates:
    18.10.2025 Initial script [Rangsiman Ketkaew]
"""

import numpy as np

from rdkit import Chem


def build_mol_from_xyz(Z, pos, bond_fuzz=0.45):
    # Construct undirected graph

    # covalent radii for common QM9 elements
    cov_radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}

    rw = Chem.RWMol()
    for z in Z:
        rw.AddAtom(Chem.Atom(int(z)))

    N = len(Z)

    # Add bonds based on distance
    edge_list = []
    for i in range(N):
        for j in range(i + 1, N):
            dij = float(np.linalg.norm(pos[i] - pos[j]))
            ri = cov_radii.get(int(Z[i]), 0.7)
            rj = cov_radii.get(int(Z[j]), 0.7)
            if dij <= ri + rj + bond_fuzz:
                try:
                    rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                except Exception:
                    pass

    mol = rw.GetMol()

    # try:
    #     Chem.SanitizeMol(mol)
    # except Exception:
    #     pass

    # partial sanitization
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
    )

    # Attach 3D conformer
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(pos):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    return mol, edge_list


def xyz_to_data(Z, pos, target):
    """
    Create dataset of graph features from Cartesian coordinates

    Args:
        Z: array of atomic numbers
        R: array of positions
        target: target energy value

    Returns:
        "x": node features from RDKit
        "edge_index": edge connectivity
        "edge_attr": edge features from RDKit
        "E": energy
    """
    # Build RDKit molecule from XYZ
    mol, edge_list = build_mol_from_xyz(Z, pos, bond_fuzz=0.45)

    # -------------------------
    # Calculate node features
    # -------------------------

    x = []

    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalValence(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            atom.GetHybridization().real,
        ]
        x.append(feat)

    x = np.array(x, dtype=np.float32)

    # -------------------------
    # Calculate edge features
    # -------------------------

    edge_attr = []
    edge_index = []

    if len(edge_list) > 0:
        edge_index = np.array(edge_list, dtype=np.int64).T

        for i, j in edge_list:
            bond = mol.GetBondBetweenAtoms(int(i), int(j))
            if bond is not None:
                feat = [
                    bond.GetBondTypeAsDouble(),
                    int(bond.GetIsConjugated()),
                    int(bond.IsInRing()),
                ]
            else:
                feat = [1.0, 0, 0]
            edge_attr.append(feat)

        edge_attr = np.array(edge_attr, dtype=np.float32)
    else:
        # no edges
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 3), dtype=np.float32)

    return {
        "Z": Z,
        "R": pos,
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "E": float(target),
    }


def qm9_to_data(qm9_data, target="U0", recal_features=False):
    """
    Create dataset from QM9 data object

    Returns:
        "x": node features from RDKit
        "edge_index": edge connectivity
        "edge_attr": edge features from RDKit
        "E": energy
    """
    Z = qm9_data.z.cpu().numpy().astype(np.int32)
    pos = qm9_data.pos.cpu().numpy().astype(np.float32)
    x = qm9_data.x.cpu().numpy().astype(np.float32)
    edge_index = qm9_data.edge_index.cpu().numpy().astype(np.float32)
    edge_attr = qm9_data.edge_attr.cpu().numpy().astype(np.float32)
    y = qm9_data.y.cpu().numpy().reshape(-1)

    # https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.QM9.html
    name_to_idx = {"U0": 7, "U": 8, "H": 9}
    idx = name_to_idx.get(target, 7)
    E = float(y[idx])

    if recal_features:
        feat = xyz_to_data(Z, pos, E)
        x = feat["x"]
        edge_index = feat["edge_index"]
        edge_attr = feat["edge_attr"]

    return {
        "Z": Z,
        "R": pos,
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "E": float(E),
    }


def qm9_splits(
    qm9,
    n_train=10000,
    n_val=1000,
    n_test=1000,
    target="U0",
    shuffle=False,
    seed=12345,
    subset=None,
):
    """Import QM9 dataset and create dataset for training a model

    # targets: QM9 stores 19 targets in qm9_data.y
    # see list of properties in torch_geometric.datasets.QM9

    Args:
        qm9_ds: QM9 dataset object from torch_geometric
        n_train: number of training examples
        n_val: number of validation examples
        n_test: number of test examples
        target: QM9 target property name (U0, U, H, etc.)
        seed: random seed for shuffling
        subset: optional limit on total dataset size

    Returns:
        train, val, test: dict of graph features
    """

    # Shuffle indices
    total = len(qm9)
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = np.arange(total)
        rng.shuffle(idx)
        qm9 = qm9[idx]

    if subset is not None:
        total = min(total, int(subset))
        qm9 = qm9[:total]
        total = len(qm9)

    n_train = min(n_train, total)
    n_val = min(n_val, max(0, total - n_train))
    n_test = min(n_test, max(0, total - n_train - n_val))

    my_data = [qm9_to_data(qm9[i], target, recal_features=True) for i in range(total)]
    train = my_data[:n_train]
    val = my_data[n_train : n_train + n_val]
    test = my_data[n_train + n_val : n_train + n_val + n_test]

    return train, val, test
