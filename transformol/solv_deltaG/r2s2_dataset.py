"""
Create solvation dataset from SMILES or XYZ csv files and save as Torch DataLoader

# Usage:

1) For SMILES csv dataset (like CombiSolv):

    > python r2s2_dataset.py --smiles-path data/CombiSolv-QM-sample-1.csv \
        --atom-dim 30 \
        --bond-dim 6 \
        --save-path data/

2) For XYZ dataset (like AQM):

    > python r2s2_dataset.py --xyz-aqm-path data/AQM-sample-1.csv  \
        --atom-dim 30 \
        --bond-dim 26 \
        --save-path data/

Note: for XYZ dataset, the RBS basis is additionally used as edge_attr feature.
The default number of RBF basis (n_gaussians) is 20. Therefore, the total number of 
bond features is 6 + 20 = 26.

Updates:
    18.10.2025 Initial script [Rangsiman Ketkaew]
"""

import argparse
import warnings
import time
import numpy as np
import pandas as pd
import torch as t

from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from sklearn.model_selection import train_test_split


warnings.filterwarnings(
    "ignore", message="not removing hydrogen atom without neighbors"
)

t.manual_seed(12345)

# fmt: off
atom_symbols = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn"
}
# fmt: on


def atom_features(atom: Chem.rdchem.Atom, atom_dim: int = 30):
    """Generate atom features for a given RDKit atom object

    - one-hot encoding for elements
    - degree, formal charge, aromatic, hybridization, implicit valence
    """
    elem = atom.GetSymbol()
    element = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H"]
    onehot = [1.0 if elem == c else 0.0 for c in element]

    # simple one-hot encoding
    if sum(onehot) == 0:
        onehot.append(1.0)
    else:
        onehot.append(0.0)

    # Degree
    degree = atom.GetDegree()
    # Formal charge
    charge = atom.GetFormalCharge()
    # is it aromatic?
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    # Hybridization
    hybrid = (
        atom.GetHybridization().real
        if hasattr(atom.GetHybridization(), "real")
        else float(atom.GetHybridization())
    )
    # Implicit valence
    implicit_valence = atom.GetTotalValence()

    feats = onehot + [
        degree,
        charge,
        aromatic,
        hybrid if hybrid < 10 else 0.0,
        implicit_valence,
    ]

    feat = np.array(feats, dtype=np.float32)

    if feat.shape[0] < atom_dim:
        feat = np.pad(feat, (0, atom_dim - feat.shape[0]), constant_values=0.0)
    else:
        feat = feat[:atom_dim]

    return feat


def bond_features(bond: Chem.rdchem.Bond, bond_dim: int = 6) -> np.ndarray:
    bt = bond.GetBondType()

    bt_onehot = [
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
    ]

    conj = 1.0 if bond.GetIsConjugated() else 0.0
    stereo = 1.0 if bond.GetStereo() != Chem.rdchem.BondStereo.STEREOANY else 0.0
    feats = bt_onehot + [conj, stereo]

    feat = np.array(feats, dtype=np.float32)

    if feat.shape[0] < bond_dim:
        feat = np.pad(feat, (0, bond_dim - feat.shape[0]), constant_values=0.0)
    else:
        feat = feat[:bond_dim]

    return feat


def smiles_to_graph(smiles: str, atom_dim: int = 30, bond_dim: int = 6):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = [atom_features(a, atom_dim) for a in mol.GetAtoms()]
    edge_index = []
    edge_attr = []

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b, bond_dim)
        # add both directions
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])

    if len(edge_index) == 0:
        # single atom molecule
        edge_index = np.zeros((2, 0), dtype=int)
        edge_attr = np.zeros((0, bond_dim), dtype=np.float32)
    else:
        edge_index = np.array(edge_index, dtype=int).T
        edge_attr = np.array(edge_attr, dtype=np.float32)

    data = Data(
        x=t.tensor(np.vstack(x), dtype=t.float32),
        edge_index=t.tensor(edge_index, dtype=t.int32),
        edge_attr=t.tensor(edge_attr, dtype=t.float32),
    )

    return data


# fmt: off
graph_cat = {}  
def get_graph(smiles: str, atom_dim: int = 30, bond_dim: int = 6):
    if smiles in graph_cat:
        return graph_cat[smiles]
    g = smiles_to_graph(smiles, atom_dim, bond_dim)
    graph_cat[smiles] = g

    return g
# fmt: on


class SolvationDataset(Dataset):
    def __init__(
        self, csv_path, atom_dim=30, bond_dim=26, transform=None, pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.df = pd.read_csv(csv_path)

        assert {"mol solute", "mol solvent", "target Gsolv kcal"}.issubset(
            set(self.df.columns)
        )

        self.samples = self.df.to_dict(orient="records")

    def len(self):
        return len(self.samples)

    def get(self, idx):
        row = self.samples[idx]
        sol_smiles = row["mol solute"]
        solv_smiles = row["mol solvent"]
        sol_graph = get_graph(sol_smiles, self.atom_dim, self.bond_dim)
        solv_graph = get_graph(solv_smiles, self.atom_dim, self.bond_dim)

        # each data can be cloned so that we can avoid in-place modifications later, like
        # x = sol_graph.x.clone()
        sol = Data(
            x=sol_graph.x,
            edge_index=sol_graph.edge_index,
            edge_attr=sol_graph.edge_attr,
        )

        solv = Data(
            x=solv_graph.x,
            edge_index=solv_graph.edge_index,
            edge_attr=solv_graph.edge_attr,
        )

        deltaG = t.tensor([row["target Gsolv kcal"]], dtype=t.float32)

        return {"solute": sol, "solvent": solv, "deltaG": deltaG}


def create_xyz_block(Z, coords):
    """
    Args:
        symbols (list): List of atomic symbols e.g., ['C', 'H', 'H', 'H', 'H']
        coordinates (np.array): N x 3 array of Cartesian coordinates

    Returns:
        str: A formatted XYZ block string
    """
    n_atoms = len(Z)
    lines = [str(n_atoms), "Generated by Python"]

    for z, coord in zip(Z, coords):
        label = atom_symbols.get(z)
        x, y, z = coord
        lines.append(f"{label:2s} {x:12.6f} {y:12.6f} {z:12.6f}")

    return "\n".join(lines)


class GaussianRBF(object):
    """Radial basis function expansion using Gaussian function for distances.
    We use RBF as interatomic distances encoding. This makes model more "3D geometric".

    Note: one need to update the bond feature dimensiion (with "--bond-dim" flag)
    when changing n_gaussians
    """

    def __init__(self, start=0.0, stop=5.0, n_gaussians=20):
        offset = np.linspace(start, stop, n_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]) ** 2
        self.offset = offset

    def __call__(self, dist):
        dist = dist[:, None] - self.offset[None, :]
        return np.exp(self.coeff * np.power(dist, 2))


rbf_expand = GaussianRBF(start=0.0, stop=5.0, n_gaussians=20)


def xyz_to_graph(Z, coords, atom_dim=30, bond_dim=26, charge=0):
    # Create RDKit mol from scratch
    # mol = Chem.RWMol()
    # for num in Z:
    #     mol.AddAtom(Chem.Atom(int(num)))

    # conf = Chem.Conformer(mol.GetNumAtoms())
    # # for i, (x, y, z) in enumerate(coords):
    # for i in range(len(coords)):
    #     x, y, z = coords[i]
    #     conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    # mol.AddConformer(conf)

    mol = Chem.MolFromXYZBlock(create_xyz_block(Z, coords))
    mol = Chem.Mol(mol)
    print("total charge of molecule:", Chem.GetFormalCharge(mol))
    # Chem.SanitizeMol(mol)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    rdDetermineBonds.DetermineConnectivity(mol)
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge)
    except ValueError:
        pass
    # rdDetermineBonds.DetermineBonds(mol, charge=charge)

    mol.UpdatePropertyCache()

    x = [atom_features(a, atom_dim) for a in mol.GetAtoms()]
    edge_index = []
    edge_attr = []
    distances = []

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b, bond_dim)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])

        pos_i = coords[i]
        pos_j = coords[j]
        dist = np.linalg.norm(pos_i - pos_j)
        distances.extend([dist, dist])

    if len(edge_index) == 0:
        edge_index = np.zeros((2, 0), dtype=int)
        edge_attr = np.zeros((0, bond_dim), dtype=np.float32)
    else:
        edge_index = np.array(edge_index, dtype=int).T
        edge_attr = np.array(edge_attr, dtype=np.float32)

        # RBF expansion
        distances = np.array(distances, dtype=np.float32)
        rbf_feat = rbf_expand(distances)
        # concatenate edge features and RBF features together
        edge_attr = np.concatenate([edge_attr, rbf_feat], axis=1)

    data = Data(
        x=t.tensor(np.vstack(x), dtype=t.float32),
        edge_index=t.tensor(edge_index, dtype=t.int32),
        edge_attr=t.tensor(edge_attr, dtype=t.float32),
    )

    return data


class SolvationDatasetFromXYZ(Dataset):
    def __init__(
        self,
        csv_path: str,
        solv_smiles: str,
        atom_dim: int = 30,
        bond_dim: int = 26,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(None, transform, pre_transform)

        df = pd.read_csv(csv_path)
        df["atomic_numbers"] = df["atomic_numbers"].apply(eval)
        df["gas_xyz"] = df["gas_xyz"].apply(eval)
        # df["sol_xyz"] = df["sol_xyz"].apply(eval)
        self.samples = df.to_dict(orient="records")
        self.solv_smiles = solv_smiles
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim

    def len(self):
        return len(self.samples)

    def get(self, idx):
        row = self.samples[idx]
        Z = np.array(row["atomic_numbers"])
        gas_xyz = np.array(row["gas_xyz"]).reshape(-1, 3)
        sol_graph = xyz_to_graph(Z, gas_xyz, self.atom_dim, self.bond_dim)
        solv_graph = get_graph(self.solv_smiles, self.atom_dim, self.bond_dim)

        deltaG = t.tensor([np.float32(row["solvation_energy"])], dtype=t.float32)

        # each data can be cloned so that we can avoid in-place modifications later, like
        # x = sol_graph.x.clone()
        sol = Data(
            x=sol_graph.x,
            edge_index=sol_graph.edge_index,
            edge_attr=sol_graph.edge_attr,
        )

        solv = Data(
            x=solv_graph.x,
            edge_index=solv_graph.edge_index,
            edge_attr=solv_graph.edge_attr,
        )

        return {"solute": sol, "solvent": solv, "deltaG": deltaG}


def collate_fn(batch):
    """return batches as lists"""
    solute, solvent, y = [], [], []

    for item in batch:
        solute.append(item["solute"])
        solvent.append(item["solvent"])
        y.append(item["deltaG"])

    return solute, solvent, t.stack(y, dim=0)


def create_dataset(
    raw_data_path: str,
    dataset_type: str,
    atom_dim: int = 30,
    bond_dim: int = 6,  # without RBF features for XYZ dataset
    batch_size: int = 32,
):

    # fmt: off
    start_time = time.time()

    if dataset_type == "smiles":
        dataset = SolvationDataset(raw_data_path, atom_dim=atom_dim, bond_dim=bond_dim)
    elif dataset_type == "xyz_aqm":
        # Since AQM uses water as solvent, solv_smiles is always water
        dataset = SolvationDatasetFromXYZ(raw_data_path, solv_smiles="O",  atom_dim=atom_dim, bond_dim=bond_dim)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    used_time = time.time()

    print(f"Featurization time: {used_time - start_time:.2f}s")

    start_time = time.time()
    idx = list(range(len(dataset)))
    train_val, test_set = train_test_split(idx, test_size=0.1, random_state=42)
    train_set, val_set = train_test_split(train_val, test_size=0.125, random_state=42)

    train_ds = Subset(dataset, train_set)
    val_ds = Subset(dataset, val_set)
    test_ds = Subset(dataset, test_set)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)
    used_time = time.time()
    print(f"Data preparation time: {used_time - start_time:.2f}s")
    # fmt: on

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Create solvation dataset and save as Torch DataLoader")
    parser.add_argument("--smiles-path", default=False, type=str, help="Path to dataset CSV file")
    parser.add_argument("--xyz-aqm-path", default=False, type=str, help="Path to dataset XYZ file for AQM dataset")
    parser.add_argument("--save-path", type=str, help="Path prefix to save DataLoader files", required=True)
    parser.add_argument("--atom-dim", type=int, default=30, help="Number of atom features")
    parser.add_argument("--bond-dim", type=int, default=6, help="Number of bond features")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    args = parser.parse_args()
    # fmt: on

    if args.smiles_path and args.xyz_aqm_path:
        raise ValueError("Please provide only one of --smiles-path or --xyz-aqm-path")
    elif args.smiles_path:
        train_loader, val_loader, test_loader = create_dataset(
            raw_data_path=args.smiles_path,
            dataset_type="smiles",
            atom_dim=args.atom_dim,
            bond_dim=args.bond_dim,
            batch_size=args.batch_size,
        )
    elif args.xyz_aqm_path:
        train_loader, val_loader, test_loader = create_dataset(
            raw_data_path=args.xyz_aqm_path,
            dataset_type="xyz_aqm",
            atom_dim=args.atom_dim,
            bond_dim=args.bond_dim,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError("Please provide one of --csv-path or --xyz-aqm-path")

    print("-" * 30)
    print("Total training samples:", len(train_loader.dataset))
    print("Total validation samples:", len(val_loader.dataset))
    print("Total test samples:", len(test_loader.dataset))
    print(
        "Total dataset samples:",
        len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
    )
    print("-" * 30)
    print("Data info of the first sample")
    print("Node dimension:", train_loader.dataset[0]["solute"].x.size(1))
    print("Edge dimension:", train_loader.dataset[0]["solute"].edge_attr.size(1))

    t.save(train_loader, f"{args.save_path}/train_loader.pth")
    t.save(val_loader, f"{args.save_path}/val_loader.pth")
    t.save(test_loader, f"{args.save_path}/test_loader.pth")
