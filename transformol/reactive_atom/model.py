"""
GNN with Pipek-Mezey surrogate orbital localization head

Updates:
    18.10.2025 Initial script [Rangsiman Ketkaew]
"""

import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import SAGEConv, GATConv
from pyscf import gto, scf, lo

from loss import localization_surrogate_J_sur, orthogonality_penalty

np.random.seed(12345)
torch.manual_seed(12345)


class MoleculeDataset(Dataset):
    """Simple dataset placeholder"""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MPNNmodel(nn.Module):
    """MPNN model"""

    def __init__(self, hidden_dim=128, n_steps=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(100, hidden_dim)  # atomic numbers up to 99
        self.feature_proj = nn.Linear(6, hidden_dim)
        self.n_steps = n_steps

        self.message = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(n_steps)]
        )
        self.update_mlps = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_steps)]
        )

    def forward(self, x, edge_index, edge_feat=None):
        if x.dim() == 1:
            h = self.embedding(x)
        else:
            h = self.feature_proj(x)

        N = h.shape[0]
        device = h.device

        if edge_index is None or edge_index.numel() == 0:
            idx = torch.arange(N, device=device)
            edge_index = torch.combinations(idx, r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        for msg_mlp, update_mlp in zip(self.message, self.update_mlps):
            src_nodes = edge_index[0]
            tgt_nodes = edge_index[1]

            h_src = h[src_nodes]
            h_tgt = h[tgt_nodes]

            edge_features = torch.cat([h_src, h_tgt], dim=-1)
            messages = F.relu(msg_mlp(edge_features))

            aggr = torch.zeros(N, self.hidden_dim, device=device)
            aggr.index_add_(0, tgt_nodes, messages)

            h = F.relu(update_mlp(aggr) + h)

        return h


class GraphSAGEmodel(nn.Module):
    """GraphSAGE model

    GraphSAGE is designed for rich node features, so doesn't use edge features.
    """

    def __init__(self, hidden_dim=128, n_steps=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(100, hidden_dim)  # atomic numbers up to 99
        self.feature_proj = nn.Linear(6, hidden_dim)
        self.n_steps = n_steps
        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(n_steps)]
        )

    def forward(self, x, edge_index, edge_feat=None):
        if x.dim() == 1:
            h = self.embedding(x)
        else:
            h = self.feature_proj(x)

        for conv in self.convs:
            x = conv(h, edge_index)
            x = F.relu(x)
            h = x

        return h


class GATmodel(nn.Module):
    """Graph Attention Network (GAT) model"""

    def __init__(self, hidden_dim=128, n_step=3, heads=4, edge_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(100, hidden_dim)  # atomic numbers up to 99
        self.feature_proj = nn.Linear(6, hidden_dim)

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim)
        )

        for _ in range(n_step - 2):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim)
            )

    def forward(self, x, edge_index, edge_feat=None):
        if x.dim() == 1:
            h = self.embedding(x)
        else:
            h = self.feature_proj(x)

        for conv in self.convs:
            h = conv(h, edge_index, edge_attr=edge_feat)
            h = F.relu(h)

        return h


class EnergyHead(nn.Module):
    """Atom-wise energy decomposition head

    - map each atom's embedding to a scalar per-atom energy via MLP
    - then sum the per-atom energies to get the predicted total energy
    """

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h):
        E_atoms = self.mlp(h).squeeze(-1)
        E = E_atoms.sum()

        return E, E_atoms


class LocalizationHead(nn.Module):
    """Produce per-orbital per-atom logits G_{iA}

    Two modes:
    1) "per_atom" outputs an (N, N_orb) matrix from which we transpose to (N_orb, N)
    2) "attention" uses learned orbital queries
    """

    def __init__(self, in_dim, n_orb=10, mode="per_atom", hidden=64):
        super().__init__()
        self.n_orb = n_orb
        self.mode = mode
        if mode == "per_atom":
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_orb)
            )
        elif mode == "attention":
            # learned orbital queries
            self.query = nn.Parameter(torch.randn(n_orb, in_dim))
            self.key = nn.Linear(in_dim, in_dim)
        else:
            raise ValueError("mode must be per_atom or attention")

    def forward(self, h):
        if self.mode == "per_atom":
            G = self.mlp(h)
            G = G.transpose(0, 1)
        else:
            K = self.key(h)
            # queries: (n_orb, in_dim) keys: (N, in_dim) => scores (n_orb, N)
            G = torch.matmul(self.query, K.t())

        return G


class MLWithLocalization(nn.Module):
    def __init__(self, model, hidden_dim=128, n_orb=10, loc_mode="per_atom"):
        super().__init__()

        if model == "mpnn":
            self.model = MPNNmodel(hidden_dim=hidden_dim)
        elif model == "sage":
            self.model = GraphSAGEmodel(hidden_dim=hidden_dim)
        elif model == "gat":
            self.model = GATmodel(hidden_dim=hidden_dim)
        else:
            exit("Available GNN models: mpnn, sage, gat")
        self.energy_head = EnergyHead(in_dim=hidden_dim)
        self.loc_head = LocalizationHead(in_dim=hidden_dim, n_orb=n_orb, mode=loc_mode)

    def forward(self, x, edge_index, edge_attr):
        h = self.model(x, edge_index, edge_attr)
        E_pred, E_atoms = self.energy_head(h)
        G = self.loc_head(h)

        # convert logits to per-orbital-per-atom normalized weights
        P = F.softmax(G, dim=1)  # softmax over atoms

        return E_pred, E_atoms, G, P


def calc_loss(model, batch_data, args):
    device = args.get("device", "cpu")

    if isinstance(batch_data, list):
        batch_data = batch_data[0]

    # Z = torch.as_tensor(batch_data["Z"], dtype=torch.long, device=device)
    R = torch.as_tensor(batch_data["R"], dtype=torch.float32, device=device)
    x = torch.as_tensor(batch_data["x"], dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(
        batch_data["edge_index"], dtype=torch.long, device=device
    )
    edge_attr = torch.as_tensor(
        batch_data["edge_attr"], dtype=torch.float32, device=device
    )
    E_ref = torch.as_tensor(batch_data["E"], dtype=torch.float32, device=device)

    # Remove batch dimension if present (DataLoader adds it with batch_size=1)
    if x.dim() == 3 and x.size(0) == 1:
        x = x.squeeze(0)
    if edge_index.dim() == 3 and edge_index.size(0) == 1:
        edge_index = edge_index.squeeze(0)
    if edge_attr.dim() == 3 and edge_attr.size(0) == 1:
        edge_attr = edge_attr.squeeze(0)
    if R.dim() == 3 and R.size(0) == 1:
        R = R.squeeze(0)

    # Make sure that E_ref is a scalar tensor (0-d) to match E_pred (sum over atoms returns 0-d)
    if hasattr(E_ref, "dim") and E_ref.dim() > 0:
        E_ref = E_ref.squeeze()

    E_pred, E_atoms, G, P = model(x, edge_index, edge_attr)

    L_E = F.mse_loss(E_pred, E_ref)
    L = L_E

    if args.get("use_forces", False) and "F" in batch_data:
        E_pred2, _, _, _ = model(x, edge_index, edge_attr)
        R_req = R.clone().detach().requires_grad_(True)
        F_pred = -torch.autograd.grad(E_pred2, R_req, create_graph=True)[0]

        F_ref = torch.as_tensor(batch_data["F"], dtype=torch.float32, device=device)
        if F_ref.dim() == 3 and F_ref.size(0) == 1:
            F_ref = F_ref.squeeze(0)

        L_F = F.mse_loss(F_pred, F_ref)
        L = L + args.get("alpha", 1.0) * L_F

    Jsur = localization_surrogate_J_sur(P)

    L = (
        L
        + args.get("beta", 0.0) * (-Jsur)
        + args.get("gamma", 0.0) * orthogonality_penalty(P)
    )

    out = {
        "loss": L,
        "L_E": L_E.detach().cpu().item(),
        "Jsur": Jsur.detach().cpu().item(),
        "E_pred": E_pred.detach().cpu().item(),
        "P": P.detach().cpu().numpy(),
    }

    return L, out


def train_epoch(model, data_loader, optimizer, args):
    model.train()
    total_loss = 0.0
    count = 0
    for batch_data in data_loader:
        optimizer.zero_grad()
        L, info = calc_loss(model, batch_data, args)
        L.backward()
        optimizer.step()
        total_loss += info["loss"]
        count += 1

    return total_loss / max(1, count)


def validate(model, val_set, device="cpu"):
    model.eval()
    E_mae = []

    with torch.no_grad():
        for batch_data in val_set:
            # Z = torch.as_tensor(batch_data["Z"], dtype=torch.long, device=device)
            # R = torch.as_tensor(batch_data["R"], dtype=torch.float32, device=device)
            x = torch.as_tensor(batch_data["x"], dtype=torch.float32, device=device)
            edge_index = torch.as_tensor(
                batch_data["edge_index"], dtype=torch.int32, device=device
            )
            edge_attr = torch.as_tensor(
                batch_data["edge_attr"], dtype=torch.float32, device=device
            )
            E_ref = float(batch_data["E"])
            E_pred, E_atoms, G, P = model(x, edge_index, edge_attr)
            # P_np = P.cpu().numpy()
            E_mae.append(abs(float(E_pred.cpu().numpy()) - E_ref))

    out = {}

    if len(E_mae) > 0:
        _out = np.mean(E_mae)
    else:
        _out = float("nan")

    out["E_mae"] = _out

    return out


def training_run(train_set, val_set, config):
    """Train a model with small localization weight and gradually increase
    to balance energy accuracy and localization quality
    """
    device = config.get("device", "cpu")

    model = MLWithLocalization(
        model=config.get("model", "mpnn"),
        hidden_dim=config.get("hidden_dim", 128),
        n_orb=config.get("n_orb", 8),
        loc_mode=config.get("loc_mode", "per_atom"),
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 5e-4))

    train_loader = DataLoader(MoleculeDataset(train_set), batch_size=1, shuffle=True)

    best_val = math.inf

    for epoch in range(config.get("n_epochs", 100)):
        # increase beta over training epochs (optional)
        beta = config.get("beta_init", 0.0) + (
            config.get("beta_final", 0.0) - config.get("beta_init", 0.0)
        ) * (epoch / (config.get("n_epochs", 1)))

        args = {
            "device": device,
            "use_forces": config.get("use_forces", False),
            "alpha": config.get("alpha", 1.0),
            "beta": beta,
            "gamma": config.get("gamma", 0.0),
        }

        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, args)
        val_metrics = validate(model, val_set, device)
        end_time = time.time() - start_time

        print(
            f"Epoch {epoch:3d} beta={beta:.4f} train_loss={train_loss:.6f} val_E_mae={val_metrics.get("E_mae", float("nan")):.6f} time={end_time:.1f}s"
        )

        if val_metrics.get("E_mae", float("inf")) < best_val:
            best_val = val_metrics["E_mae"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                config.get("checkpoint_path", "best_ckpt.pt"),
            )

    check_point = torch.load(
        config.get("checkpoint_path", "best_ckpt.pt"), map_location=device
    )

    model.load_state_dict(check_point["model_state"])

    final_val = validate(
        model,
        val_set,
        device,
    )

    return model, final_val


def predict(
    model,
    test_set,
    device="cpu",
    reactive_mode="sum",
    threshold=None,
):
    """Predict energy and rank atoms using score-based ranking using reactive_mode aggregation over orbitals (sum/max/mean)

    Reactive atom ranking: aggregate per-orbital weights into per-atom scores

    $s_A = \mathrm{agg}_i P_{iA}$,

    where agg is sum/max/mean rank atoms descending by $s_A$.

    Returns:
        E_pred: predicted energy
        reactive_indices: reactive atom indices
        reactive_ranking_indices: all atoms sorted by ranking metric
        atom_scores: score-based aggregation used for fallback or alongside AUC
    """
    model.eval()
    results = []

    with torch.no_grad():
        for item in test_set:
            Z = torch.as_tensor(item["Z"], dtype=torch.long, device=device)
            R = torch.as_tensor(item["R"], dtype=torch.float32, device=device)
            x = torch.as_tensor(item["x"], dtype=torch.float32, device=device)
            edge_index = torch.as_tensor(
                item["edge_index"], dtype=torch.int32, device=device
            )
            edge_attr = torch.as_tensor(
                item["edge_attr"], dtype=torch.float32, device=device
            )
            # E_ref = torch.as_tensor(item["E"], dtype=torch.float32, device=device)

            E_pred, E_atoms, G, P = model(x, edge_index, edge_attr)
            E_pred_scalar = float(E_pred.detach().cpu().numpy())

            P_np = P.detach().cpu().numpy()  # (n_orb, N)

            reactive_ranking_indices = []

            if reactive_mode == "sum":
                atom_scores = P_np.sum(axis=0)
            elif reactive_mode == "max":
                atom_scores = P_np.max(axis=0)
            elif reactive_mode == "mean":
                atom_scores = P_np.mean(axis=0)
            else:
                raise ValueError("Available reactive_mode: sum, max, mean")

            reactive_ranking_indices = np.argsort(-atom_scores).tolist()

            if threshold is not None:
                # so, let's use thresholding only to score-based path; for AUC, just use top_k
                reactive_indices = np.where(atom_scores >= float(threshold))[0].tolist()
                if len(reactive_indices) == 0:
                    reactive_indices = reactive_ranking_indices
            else:
                reactive_indices = reactive_ranking_indices

            results.append(
                {
                    "E_pred": E_pred_scalar,
                    "reactive_indices": reactive_indices,
                    "reactive_ranking_indices": reactive_ranking_indices,
                    "atom_scores": atom_scores.astype(float),
                }
            )

    return results


def calc_pm_populations(atomic_numbers, xyz):
    """
    Perform a simple HF and Pipek-Mezey (PM) localization

    Args:
        atomic_numbers: Atomic numbers
        xyz: Atomic coordinates

    Returns:
        Q: Mulliken population fractions per orbital (n_occ, N_atoms)
        J_PM: the Pipek-Mezey objective
    """

    natoms = len(atomic_numbers)
    atom_str = []

    for Z, coord in zip(atomic_numbers, xyz.tolist()):
        atom_str.append((Z, tuple(coord)))

    mol = gto.M(atom=atom_str, basis="sto-3g", charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    mo_coeff = mf.mo_coeff  # coefficients in AO basis

    loc = lo.PM(mol, mo_coeff)
    mo_loc = loc.kernel()
    mo_loc = np.asarray(mo_loc)

    if np.iscomplexobj(mo_loc):
        mo_loc = mo_loc.real

    # calculate Mulliken populations per orbital/atom
    S = mol.intor("int1e_ovlp")
    S = 0.5 * (S + S.T)

    nocc = mol.nelectron // 2
    Q = np.zeros((nocc, natoms))

    # Mapping from AO index to atom index
    ao_labels = mol.ao_labels()

    ao_to_atom = []
    for lab in ao_labels:
        if isinstance(lab, tuple) and len(lab) > 0:
            ao_to_atom.append(int(lab[0]))
        else:
            ao_to_atom.append(int(str(lab).split()[0]))

    for i in range(nocc):
        Ci = mo_loc[:, i]
        for A in range(natoms):
            idxs = [j for j, a_idx in enumerate(ao_to_atom) if int(a_idx) == A]

            if len(idxs) == 0:
                Q[i, A] = 0.0
                continue

            CiA = Ci[idxs]
            S_A = S[np.ix_(idxs, idxs)]
            qA = CiA.T @ S_A @ CiA
            Q[i, A] = qA

    # normalize
    row_sums = Q.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    Q_norm = Q / row_sums
    J_PM = np.sum(Q_norm**2, dtype=np.float32)

    return Q_norm, J_PM


def pm_atom_ranking(Q_norm, mode="pm"):
    """Rank atoms using only Pipek-Mezey populations

    Args:
        Q_norm: Normalized PM populations (num occ, num atoms)
        mode: "pm" (Pipek-Mezey contribution), "sum", "max"

    Returns list of atom indices sorted by descending score
    """
    if mode == "pm":
        score = (Q_norm**2).sum(axis=0)
    elif mode == "sum":
        score = Q_norm.sum(axis=0)
    elif mode == "max":
        score = Q_norm.max(axis=0)
    else:
        raise ValueError("Available mode: pm, sum, max")

    order = np.argsort(-score).tolist()

    return order
