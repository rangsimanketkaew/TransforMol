import pandas as pd
import torch as t

from torch_geometric.data import Data, Dataset
from rdkit import Chem


def smiles_to_graph(smiles):
    """
    Convert SMILES string to graph representation with node and edge features.

    Node features (9 dimensions):
    - Atomic number (one-hot for common elements: C, N, O, S, F, Cl, Br, other)
    - Degree
    - Formal charge
    - Hybridization (one-hot: SP, SP2, SP3, other)
    - Aromaticity
    - Number of hydrogens

    Edge features (4 dimensions):
    - Bond type (one-hot: single, double, triple, aromatic)
    """
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.MolFromXYZFile(xyz)
    if mol is None:
        return None

    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        # Atomic number (simplified one-hot)
        # C, N, O, S, F, Cl, Br
        atom_num = atom.GetAtomicNum()
        atom_encoding = [0] * 8
        if atom_num == 6:
            atom_encoding[0] = 1
        elif atom_num == 7:
            atom_encoding[1] = 1
        elif atom_num == 8:
            atom_encoding[2] = 1
        elif atom_num == 16:
            atom_encoding[3] = 1
        elif atom_num == 9:
            atom_encoding[4] = 1
        elif atom_num == 17:
            atom_encoding[5] = 1
        elif atom_num == 35:
            atom_encoding[6] = 1
        else:
            atom_encoding[7] = 1

        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()

        hybridization = atom.GetHybridization()
        hybrid_encoding = [0] * 4
        if hybridization == Chem.HybridizationType.SP:
            hybrid_encoding[0] = 1
        elif hybridization == Chem.HybridizationType.SP2:
            hybrid_encoding[1] = 1
        elif hybridization == Chem.HybridizationType.SP3:
            hybrid_encoding[2] = 1
        else:
            hybrid_encoding[3] = 1

        is_aromatic = int(atom.GetIsAromatic())
        num_hs = atom.GetTotalNumHs()

        # feature = atom_encoding + [degree, formal_charge] + hybrid_encoding + [is_aromatic, num_hs]
        feature = [degree, formal_charge] + hybrid_encoding + [is_aromatic, num_hs]
        node_features.append(feature)

    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Add both directions for undirected graph
        edge_indices.append([i, j])
        edge_indices.append([j, i])

        bond_type = bond.GetBondType()
        bond_enc = [0] * 4
        if bond_type == Chem.BondType.SINGLE:
            bond_enc[0] = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_enc[1] = 1
        elif bond_type == Chem.BondType.TRIPLE:
            bond_enc[2] = 1
        elif bond_type == Chem.BondType.AROMATIC:
            bond_enc[3] = 1

        # use the same features for both directions
        edge_features.append(bond_enc)
        edge_features.append(bond_enc)

    x = t.tensor(node_features, dtype=t.float)
    edge_index = (
        t.tensor(edge_indices, dtype=t.long).t().contiguous()
        if edge_indices
        else t.empty((2, 0), dtype=t.long)
    )
    edge_attr = (
        t.tensor(edge_features, dtype=t.float)
        if edge_features
        else t.empty((0, 4), dtype=t.float)
    )

    return x, edge_index, edge_attr


def csv_to_data(csv):
    """
    Convert a SMILES file (solute SMILES, dielectric constant, free energy) to a PyTorch Geometric dataset.

    Returns a list of Data objects.
    """
    df = pd.read_csv(csv)
    data_list = []

    for idx, row in df.iterrows():
        solute_smiles = row.iloc[0]
        dielectric = row.iloc[1]
        free_energy = row.iloc[2]

        solute_graph = smiles_to_graph(solute_smiles)
        if solute_graph is None:
            continue

        solute_x, solute_edge_index, solute_edge_attr = solute_graph

        # Create tensors
        y = t.tensor([free_energy], dtype=t.float)
        dielectric_tensor = t.tensor([dielectric], dtype=t.float)

        # Create Data object with dielectric as additional feature
        data = Data(
            x=solute_x,
            edge_index=solute_edge_index,
            edge_attr=solute_edge_attr,
            y=y,
            dielectric=dielectric_tensor,
        )
        data_list.append(data)

    return data_list


class SolvationDataset(Dataset):
    """Custom dataset for solvation free energy prediction."""

    def __init__(self, csv_file):
        super(SolvationDataset, self).__init__()
        self.data_df = pd.read_csv(csv_file)
        self.processed_data = []

        for idx in range(len(self.data_df)):
            solute_smiles = self.data_df.iloc[idx, 1]
            solvent_smiles = self.data_df.iloc[idx, 0]
            free_energy = self.data_df.iloc[idx, 2]

            solute_graph = smiles_to_graph(solute_smiles)
            solvent_graph = smiles_to_graph(solvent_smiles)

            if solute_graph is not None and solvent_graph is not None:
                self.processed_data.append(
                    {"solute": solute_graph, "solvent": solvent_graph, "y": free_energy}
                )

    def len(self):
        return len(self.processed_data)

    def get(self, idx):
        item = self.processed_data[idx]

        solute_x, solute_edge_index, solute_edge_attr = item["solute"]
        solvent_x, solvent_edge_index, solvent_edge_attr = item["solvent"]

        # Combine solute and solvent into a single graph
        # Offset solvent node indices
        num_solute_nodes = solute_x.size(0)

        # Handle empty edge cases
        if solvent_edge_index.size(1) > 0:
            solvent_edge_index_offset = solvent_edge_index + num_solute_nodes
        else:
            solvent_edge_index_offset = solvent_edge_index

        x = t.cat([solute_x, solvent_x], dim=0)

        if solute_edge_index.size(1) > 0 and solvent_edge_index.size(1) > 0:
            edge_index = t.cat([solute_edge_index, solvent_edge_index_offset], dim=1)
            edge_attr = t.cat([solute_edge_attr, solvent_edge_attr], dim=0)
        elif solute_edge_index.size(1) > 0:
            edge_index = solute_edge_index
            edge_attr = solute_edge_attr
        elif solvent_edge_index.size(1) > 0:
            edge_index = solvent_edge_index_offset
            edge_attr = solvent_edge_attr
        else:
            edge_index = t.empty((2, 0), dtype=t.long)
            edge_attr = t.empty((0, 4), dtype=t.float)

        # Create batch indicator for pooling
        batch = t.cat(
            [
                t.zeros(num_solute_nodes, dtype=t.long),
                t.ones(solvent_x.size(0), dtype=t.long),
            ]
        )

        y = t.tensor([item["y"]], dtype=t.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch)

        return data
