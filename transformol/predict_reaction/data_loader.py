"""
Data loading and featurization module for reaction prediction.
This module handles loading .xyz files and creating PyTorch Geometric datasets.

Updates:
    14.10.2025 Initial script [Rangsiman Ketkaew]
"""

import os
import numpy as np
import torch
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from rdkit import Chem
from typing import List, Tuple, Dict


def read_xyz(filepath: str) -> Tuple[List[str], np.ndarray]:
    with open(filepath, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    atoms, coords = [], []

    for i in range(2, 2 + n_atoms):
        parts = lines[i].strip().split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return atoms, np.array(coords)


def determine_bonds(mol, atoms, coords) -> Chem.RWMol:
    # Covalent radii in Angstroms
    covalent_radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "P": 1.07,
        "S": 1.05,
        "Cl": 1.02,
        "Br": 1.20,
        "I": 1.39,
        "B": 0.84,
        "Si": 1.11,
    }

    tolerance = 1.3

    n_atoms = len(atoms)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(coords[i] - coords[j])

            r1 = covalent_radii.get(atoms[i], 0.77)
            r2 = covalent_radii.get(atoms[j], 0.77)

            if distance < (r1 + r2) * tolerance:
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(mol)
    except:
        pass

    return mol


def xyz_to_mol(atoms: List[str], coords: np.ndarray) -> Chem.Mol:
    mol = Chem.RWMol()
    for atom_symbol in atoms:
        atom = Chem.Atom(atom_symbol)
        mol.AddAtom(atom)

    conf = Chem.Conformer(len(atoms))
    for i, coord in enumerate(coords):
        conf.SetAtomPosition(i, coord.tolist())
    mol.AddConformer(conf)
    mol = determine_bonds(mol, atoms, coords)

    return mol.GetMol()


def get_atom_features(atom) -> np.ndarray:
    atom_types = ["C", "N", "O", "H", "F", "S", "Cl", "Br", "P", "I"]
    atom_type_enc = [int(atom.GetSymbol() == x) for x in atom_types]

    degree_enc = [int(atom.GetDegree() == x) for x in range(6)]

    formal_charge = atom.GetFormalCharge()

    hybridizations = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    ]
    hybridization_enc = [int(atom.GetHybridization() == x) for x in hybridizations]

    aromatic = [int(atom.GetIsAromatic())]

    num_hs = [atom.GetTotalNumHs()]

    valence = [atom.GetTotalValence()]

    features = (
        atom_type_enc
        + degree_enc
        + [formal_charge]
        + hybridization_enc
        + aromatic
        + num_hs
        + valence
    )

    return np.array(features, dtype=np.float32)


def get_bond_features(bond) -> np.ndarray:
    bond_types = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]
    bond_type_enc = [int(bond.GetBondType() == x) for x in bond_types]

    conjugated = [int(bond.GetIsConjugated())]
    in_ring = [int(bond.IsInRing())]

    features = bond_type_enc + conjugated + in_ring

    return np.array(features, dtype=np.float32)


def mol_to_graph(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_atoms = mol.GetNumAtoms()

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    node_features = np.array(node_features, dtype=np.float32)

    edge_index = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_feat = get_bond_features(bond)
        edge_index.append([i, j])
        edge_features.append(edge_feat)
        edge_index.append([j, i])
        edge_features.append(edge_feat)

    if len(edge_index) > 0:
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_features = np.array(edge_features, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, 6), dtype=np.float32)

    return node_features, edge_index, edge_features


class ReactionDataset(TorchDataset):
    def __init__(
        self, data_dir: str, reaction_ids: List[int], transform=None, pre_transform=None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.reaction_ids = reaction_ids
        self.transform = transform
        self.pre_transform = pre_transform
        self.data_list = self._process()

    def _process(self) -> List[Dict]:
        """Process all reactions and create dataset."""
        data_list = []

        print(f"Processing {len(self.reaction_ids)} reactions...")

        for rxn_id in tqdm(self.reaction_ids):
            try:
                r_path = os.path.join(self.data_dir, f"r{rxn_id:06d}.xyz")
                ts_path = os.path.join(self.data_dir, f"ts{rxn_id:06d}.xyz")
                p_path = os.path.join(self.data_dir, f"p{rxn_id:06d}.xyz")

                if not all(os.path.exists(p) for p in [r_path, ts_path, p_path]):
                    continue

                r_atoms, r_coords = read_xyz(r_path)
                ts_atoms, ts_coords = read_xyz(ts_path)
                p_atoms, p_coords = read_xyz(p_path)

                r_mol = xyz_to_mol(r_atoms, r_coords)
                ts_mol = xyz_to_mol(ts_atoms, ts_coords)
                p_mol = xyz_to_mol(p_atoms, p_coords)

                r_node_feat, r_edge_idx, r_edge_feat = mol_to_graph(r_mol)
                ts_node_feat, ts_edge_idx, ts_edge_feat = mol_to_graph(ts_mol)
                p_node_feat, p_edge_idx, p_edge_feat = mol_to_graph(p_mol)

                # here create dataset object with attributes x, edge_index, edge_attr, pos
                r_data = Data(
                    x=torch.from_numpy(r_node_feat),
                    edge_index=torch.from_numpy(r_edge_idx),
                    edge_attr=torch.from_numpy(r_edge_feat),
                    pos=torch.from_numpy(r_coords).float(),
                )

                ts_data = Data(
                    x=torch.from_numpy(ts_node_feat),
                    edge_index=torch.from_numpy(ts_edge_idx),
                    edge_attr=torch.from_numpy(ts_edge_feat),
                    pos=torch.from_numpy(ts_coords).float(),
                )

                p_data = Data(
                    x=torch.from_numpy(p_node_feat),
                    edge_index=torch.from_numpy(p_edge_idx),
                    edge_attr=torch.from_numpy(p_edge_feat),
                    pos=torch.from_numpy(p_coords).float(),
                )

                data_list.append(
                    {
                        "rxn_id": rxn_id,
                        "reactant": r_data,
                        "ts": ts_data,
                        "product": p_data,
                    }
                )

            except Exception as e:
                print(f"Error processing reaction {rxn_id}: {e}")
                continue

        return data_list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Dict:
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict:
        return self.data_list[idx]


def create_datasets(
    data_dir: str,
    n_reactions: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    save_dir: str = "./datasets",
):

    reaction_ids = list(range(1, n_reactions + 1))
    np.random.seed(42)
    np.random.shuffle(reaction_ids)
    n_train = int(len(reaction_ids) * train_ratio)
    n_val = int(len(reaction_ids) * val_ratio)

    train_ids = reaction_ids[:n_train]
    val_ids = reaction_ids[n_train : n_train + n_val]
    test_ids = reaction_ids[n_train + n_val :]
    train_dataset = ReactionDataset(data_dir, train_ids)
    val_dataset = ReactionDataset(data_dir, val_ids)
    test_dataset = ReactionDataset(data_dir, test_ids)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "train_dataset.pkl"), "wb") as f:
        pickle.dump(train_dataset, f)

    with open(os.path.join(save_dir, "val_dataset.pkl"), "wb") as f:
        pickle.dump(val_dataset, f)

    with open(os.path.join(save_dir, "test_dataset.pkl"), "wb") as f:
        pickle.dump(test_dataset, f)

    return train_dataset, val_dataset, test_dataset


def load_datasets(save_dir: str = "./datasets"):
    with open(os.path.join(save_dir, "train_dataset.pkl"), "rb") as f:
        train_dataset = pickle.load(f)

    with open(os.path.join(save_dir, "val_dataset.pkl"), "rb") as f:
        val_dataset = pickle.load(f)

    with open(os.path.join(save_dir, "test_dataset.pkl"), "rb") as f:
        test_dataset = pickle.load(f)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # fmt: off
    data_dir = "/home/cds/rketkaew/dataset/Grambow-R-TS-P-dataset/wb97xd3_xyz/"
    train_ds, val_ds, test_ds = create_datasets(data_dir=data_dir, n_reactions=100, save_dir="./datasets")

    print(f"Dataset Statistics:")
    print(f" Train size : {len(train_ds)}")
    print(f" Val size   : {len(val_ds)}")
    print(f" Test size  : {len(test_ds)}")
    if len(train_ds) > 0:
        sample = train_ds.get(0)
        print(f"\nExample reaction {sample["rxn_id"]}:")
        print(f" Reactant: {sample["reactant"].x.shape[0]} atoms, {sample["reactant"].edge_index.shape[1]} edges")
        print(f" TS: {sample["ts"].x.shape[0]} atoms, {sample["ts"].edge_index.shape[1]} edges")
        print(f" Product: {sample["product"].x.shape[0]} atoms, {sample["product"].edge_index.shape[1]} edges")
        print(f" Node feature dim: {sample["reactant"].x.shape[1]}")
        print(f" Edge feature dim: {sample["reactant"].edge_attr.shape[1]}")
    # fmt: on
