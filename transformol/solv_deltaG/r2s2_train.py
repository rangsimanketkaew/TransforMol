"""
r2s2-GAT model for solvation free energy prediction

Usage: 

> python r2s2_train.py --train-set train_loader.pth --val-set val_loader.pth \
    [--test-set test_loader.pth] \
    [--atom-dim 30] \
    [--bond-dim 26] \
    [--epochs 10] \
    [--save-model pre-trained-model] \
    [--device cpu]

Note: bond dimension depends on the number of bond features used in RBF expansion.

Updates:
    18.10.2025 Initial script [Rangsiman Ketkaew]
"""

import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from typing import List

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from r2s2_dataset import SolvationDataset, SolvationDatasetFromXYZ, collate_fn

np.random.seed(12345)
t.manual_seed(12345)


class SolventEmbMLP(nn.Module):
    """
    MLP-based solvent embedding with attention pooling to get solv_g.

    Returns:
        solv_h, solvent node embeddings (solv_h is `u_{p}` in equations`)
        solv_g, solvent graph embedding (solvent global-level feature vector)
    """

    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()

        self.layer_1 = nn.Linear(in_dim, hidden_dim)
        self.layer_2a = nn.Linear(hidden_dim, hidden_dim)
        self.layer_2b = nn.Linear(hidden_dim + 1, hidden_dim)

        self.pool_att = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, solv_h, global_feat=None):
        solv_h = self.layer_1(solv_h)
        solv_h = F.relu(solv_h)

        if global_feat is not None:
            global_feat = global_feat.repeat(solv_h.size(0), 1)
            solv_h = t.cat([solv_h, global_feat], dim=1)
            solv_h = self.layer_2b(solv_h)
        else:
            solv_h = self.layer_2a(solv_h)
        solv_h = F.relu(solv_h)
        logits = self.pool_att(solv_h)
        attn_solv_h = F.softmax(logits, dim=0)
        solv_g = (attn_solv_h * solv_h).sum(dim=0)

        return solv_h, solv_g


class SolventEmbGAT(nn.Module):
    """
    GAT-based solvent embedding with attention pooling to get solv_g.

    Returns:
        solv_h, solvent node embeddings (solv_h is `u_{p}` in equations`)
        solv_g, solvent graph embedding (solvent global-level feature vector)
    """

    def __init__(self, in_dim, hidden_dim=128, n_layers=2, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, concat=False))
        for _ in range(1, n_layers):
            self.layers.append(
                GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
            )

        self.pool_att = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, solv_h, edge_index, edge_attr):
        for gat in self.layers:
            if not edge_attr:
                solv_h = F.relu(gat(solv_h, edge_index, edge_attr))
            else:
                solv_h = F.relu(gat(solv_h, edge_index))

        logits = self.pool_att(solv_h)
        attn_solv_h = F.softmax(logits, dim=0)
        solv_g = (attn_solv_h * solv_h).sum(dim=0)

        return solv_h, solv_g


def scatter_sum(src, index, dim_size=None):
    """scatter softmax and sum"""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out = src.new_zeros((dim_size, src.size(1)))
    out.index_add_(0, index, src)

    return out


def softmax_edge(src, index, N):
    """Calculate softmax of src grouped by index using torch.nn.functional.softmax"""
    output = src.new_zeros(src.size())

    for i in range(N):
        mask = index == i
        if mask.any():
            output[mask] = F.softmax(src[mask], dim=0)

    return output


class SoluteGATLayer(nn.Module):
    """
    Receive-and-Response layer (single layer) with solvent conditioning via FiLM and cross-attention

    FiLM-like gamma for each head: map solvent vector to a multiplicative vector
    Feature-wise Linear Modulation (FiLM) inspired by Perez et al., 2018's work

    Aggregator used in this layer combines the following techniques:
        1. multi-head attention
        2. learned per-edge modulation (edge-type gate)
        3. head-importance gating mechanism (learned soft weighting of heads per node)

    Node is updated with hidden state via Gated Message Unit (GMU) (inspired by GRU).
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        n_heads=4,
        edge_dim=6,
        n_rbf_features=20,
        solvent_dim=128,
        K_tokens=4,
        lambda_gmu=0.1,
        device="cpu",
    ):
        super().__init__()
        self.heads = n_heads
        self.n_rbf_features = n_rbf_features
        self.K_tokens = K_tokens
        self.d_h = out_dim // self.heads  # head dimension
        self.device = device

        assert (
            self.d_h * self.heads == out_dim
        ), "Error: out_dim must be divisible by n_heads"

        # ---------------------
        # Prepare ingredients
        # ---------------------
        # linear projections per head
        self.proj_k = nn.ModuleList(
            [nn.Linear(in_dim, self.d_h) for _ in range(self.heads)]
        )
        self.proj_q = nn.ModuleList(
            [nn.Linear(in_dim, self.d_h) for _ in range(self.heads)]
        )
        self.proj_m = nn.ModuleList(
            [nn.Linear(in_dim, self.d_h) for _ in range(self.heads)]
        )
        self.B_e = nn.ModuleList(
            [nn.Linear(edge_dim, self.d_h) for _ in range(self.heads)]
        )

        self.gamma = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(solvent_dim, self.d_h), nn.Tanh())
                for _ in range(self.heads)
            ]
        )

        self.g_edge = nn.ModuleList(
            [nn.Linear(self.d_h + solvent_dim + edge_dim, 1) for _ in range(self.heads)]
        )

        self.W_beta = nn.Linear(in_dim + solvent_dim, self.heads)
        self.r_h = nn.Parameter(t.randn(self.heads, self.d_h))
        self.c_h = nn.Parameter(t.zeros(self.heads))
        self.logits = nn.Linear(self.heads, self.heads, bias=False)

        self.out_lin = nn.Linear(self.d_h, out_dim)

        self.cross_q = nn.Linear(in_dim, self.d_h * self.heads)
        self.cross_k = nn.Linear(in_dim, self.d_h * self.K_tokens)
        self.cross_v = nn.Linear(in_dim, self.d_h * self.K_tokens)
        self.cross_out = nn.Linear(self.d_h * self.K_tokens, out_dim)

        # GMU parameters
        self.W_z = nn.Linear(in_dim + out_dim + solvent_dim, in_dim)
        self.W_r = nn.Linear(in_dim + out_dim + solvent_dim, in_dim)
        self.W_h = nn.Linear(in_dim + out_dim + solvent_dim, in_dim)
        self.lambda_res = nn.Parameter(t.tensor(lambda_gmu))

        # Add a small MLP to project distance to a scalar bias
        self.dist_bias_mlp = nn.Sequential(
            nn.Linear(self.n_rbf_features, 32),
            nn.ReLU(),
            nn.Linear(32, self.heads),  # One bias per head
        )

    def forward(self, x, edge_index, edge_attr, solv_g, solvent_nodes=None):
        N = x.size(0)
        row, col = edge_index

        # ------------------------------------------------------
        # Pre-calculate geometric bias for all heads
        # ------------------------------------------------------
        # Extract RBF features from the end of edge_attr
        rbf_feat = edge_attr[:, -self.n_rbf_features :]
        # # Calculate geometric bias: [num_edges, n_heads]
        geo_bias_all = self.dist_bias_mlp(rbf_feat)

        # ------------------------------------------------------
        # Step 1: Calculate multi-head messages per node
        # ------------------------------------------------------
        # 1.1 calculate per-head messages and aggregate them
        head_msgs = []
        for h in range(self.heads):
            k_i = self.proj_k[h](x)
            q_j = self.proj_q[h](x)
            m_j = self.proj_m[h](x)

            pw_compat = (k_i[row] * q_j[col]).sum(dim=-1) / np.sqrt(self.d_h)

            # solvent conditioning
            m_j_nei = m_j[col]
            m_ij_h = m_j_nei + self.B_e[h](edge_attr)
            gamma_h = self.gamma[h](solv_g)
            solv_bias = (gamma_h * F.tanh(m_ij_h)).sum(dim=-1)

            # Add to attention scores
            # Select the bias corresponding to the current head 'h'
            # attn_raw = pw_compat + solv_bias
            attn_raw = pw_compat + solv_bias + geo_bias_all[:, h]

            # normalize attention across neighbors per target node (a_{ij,h})
            attn = softmax_edge(attn_raw, row, N)

            solv_g_edge = solv_g.unsqueeze(0).expand(edge_attr.size(0), -1)
            g_in = t.cat([m_ij_h, solv_g_edge, edge_attr], dim=-1)
            g_val = F.sigmoid(self.g_edge[h](g_in)).squeeze(-1)

            # then aggregate messages per head (m^{agg}_{i,h}) by computing weighted sum
            m_weight = (attn * g_val).unsqueeze(-1) * m_ij_h
            m_agg = scatter_sum(m_weight, row, dim_size=N)
            head_msgs.append(m_agg)

        # 1.2 combine heads with importance weighting
        head_msgs = t.stack(head_msgs, dim=0)
        # where H = number of heads, then rearrange head messages so that it can multiply with beta
        head_msgs = head_msgs.permute(1, 0, 2)

        # 1.3 calculate head importance weights (beta)
        cat_in = t.cat([x, solv_g.expand(N, -1)], dim=-1)
        beta_logits = self.logits(F.tanh(self.W_beta(cat_in)))
        beta = F.softmax(beta_logits, dim=-1).unsqueeze(-1)

        combined = (beta * head_msgs).sum(dim=1)
        combined_out = self.out_lin(combined)

        # --------------------------------------
        # Step 2: Cross-attend solute to solvent
        # --------------------------------------
        if solvent_nodes is not None:
            # -------------------
            # Method 1) Cross-attention btw N solute nodes with M solvent nodes
            # -------------------
            Q = self.cross_q(x)
            Kt = self.cross_k(solvent_nodes)
            Vt = self.cross_v(solvent_nodes)
            attn_logits = t.matmul(Q, Kt.T) / np.sqrt(Q.size(-1))
            attn_weight = F.softmax(attn_logits, dim=-1)
            cross_attn = t.matmul(attn_weight, Vt)

            # --------------------------------------
            # Method 2) Cross-attention between N solute nodes with K tokens created from M solvent nodes
            # --------------------------------------
            # Q = self.cross_q(x)
            # w_logits = self.cross_k(solvent_nodes)
            # zeta = F.softmax(w_logits, dim=0)
            # tks = t.einsum("pk,pd->kd", zeta, solvent_nodes)
            # Vt = self.cross_v(tks)
            # attn_logits = t.matmul(Q, tks.T) / np.sqrt(Q.size(-1))
            # attn_weight = F.softmax(attn_logits, dim=-1)
            # cross_attn = t.matmul(attn_weight, Vt)

            # // optional to above manual implementation: use PyTorch built-in function for multiplicative attention
            # backend = SDPBackend.FLASH_ATTENTION if self.device == "cuda" else SDPBackend.MATH
            # with sdpa_kernel(backend):
            #     cross_attn = F.scaled_dot_product_attention(Q, Kt, Vt, dropout_p=0.0)

            cross_attn_vec = self.cross_out(cross_attn)
            combined_out = combined_out + cross_attn_vec

        # ------------------------------
        # Step 3: Update node with GMU
        # ------------------------------
        concat_in = t.cat([x, combined_out, solv_g.expand(N, -1)], dim=-1)
        z = F.sigmoid(self.W_z(concat_in))
        r = F.sigmoid(self.W_r(concat_in))

        rh = r * x
        concat_in = t.cat([rh, combined_out, solv_g.expand(x.size(0), -1)], dim=-1)
        h = F.tanh(self.W_h(concat_in))

        # update x^{l}_{i} to x^{l+1}_{i}
        x_updated = ((1 - z) * x) + (z * h) + (self.lambda_res * x)

        return x_updated, attn_weight


class R2S2Pool(nn.Module):
    """Graph pooling with receive-and-response (solvent-conditioned) attention

    Calculate attention-weighted mean, max, and variance of node features
    then concatenate with solvent graph-level feature solv_g.
    """

    def __init__(self, node_dim, solvent_dim, hidden=256):
        super().__init__()
        self.Wp = nn.Linear(node_dim + solvent_dim, hidden)
        self.u = nn.Linear(hidden, 1)

    def forward(self, node_states, solv_g):
        N = node_states.size(0)
        solv_g_exp = solv_g.expand(N, -1)
        concat = t.cat([node_states, solv_g_exp], dim=-1)
        logits = self.u(F.tanh(self.Wp(concat))).squeeze(-1)
        w = F.softmax(logits, dim=0).unsqueeze(-1)

        mean_w = (w * node_states).sum(dim=0)
        max_w = (w * node_states).max(dim=0)[0]
        var_w = (w * (node_states - mean_w) ** 2).sum(dim=0)

        fp = t.cat([mean_w, max_w, var_w, solv_g], dim=-1)

        return fp


class R2S2GATModel(nn.Module):
    def __init__(
        self,
        atom_in=30,
        edge_in=6,
        hidden=128,
        n_heads=4,
        n_layers=3,
        K_tokens=4,
        lambda_gmu=0.1,
        device="cpu",
    ):
        super().__init__()
        # --------
        # Solvent graph
        # --------
        self.solv_emb = SolventEmbMLP(in_dim=atom_in, hidden_dim=hidden)
        # self.solv_emb = SolventEmbGAT(in_dim=atom_in, hidden_dim=hidden, heads=n_heads)

        # -----------
        # Solute GAT
        # -----------
        self.input_proj = nn.Linear(atom_in, hidden)

        self.layers = nn.ModuleList(
            [
                SoluteGATLayer(
                    in_dim=hidden,
                    out_dim=hidden,
                    n_heads=n_heads,
                    edge_dim=edge_in,
                    solvent_dim=hidden,
                    K_tokens=K_tokens,
                    lambda_gmu=lambda_gmu,
                    device=device,
                )
                for _ in range(n_layers)
            ]
        )

        # ---------
        # Pooling
        # ---------
        self.pool = R2S2Pool(node_dim=hidden, solvent_dim=hidden)

        # Regressor: mu + max + var + solv_g
        fp_dim = hidden + hidden + hidden + hidden
        self.reg = nn.Sequential(
            nn.Linear(fp_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward_single(self, sol, solv):
        # ----------------------------------------
        # Compose a graph from solute and solvent
        # ----------------------------------------

        solv_h, solv_g = self.solv_emb(solv.x)
        # solv_h, solv_g = self.solv_emb(solv.x, solv.edge_index, solv.edge_attr)

        # Process solute with solvent conditioning
        x = F.relu(self.input_proj(sol.x))
        attn_weight = None
        for layer in self.layers:
            x, attn_weight = layer(
                x, sol.edge_index[0:2], sol.edge_attr, solv_g, solvent_nodes=solv_h
            )

        f = self.pool(x, solv_g)

        return self.reg(f).squeeze(-1), attn_weight

    def forward(self, sol_list: List[Data], solv_list: List[Data]):
        outs = []
        attn_weights = []
        for sol, solv in zip(sol_list, solv_list):
            out, attn_weight = self.forward_single(sol, solv)
            outs.append(out)
            attn_weights.append(attn_weight)

        return t.stack(outs, dim=0), attn_weights


def train(model, train_loader, optimizer, device="cpu"):
    """Train the model for one epoch

    Args:
        model: R2S2GATModel
        train_loader: Training DataLoader
        optimizer: Optimizer
        device: device to use

    Returns:
        Training loss
    """
    model.train()
    total_loss = 0.0

    # For performance, we move batch to device in each epoch
    # However, we can also move the solute and solvent datasets
    # to device in `collate_fn` if the they fit in memory.

    for solutes, solvents, y in tqdm(train_loader, desc="train"):
        optimizer.zero_grad()

        for s in solutes:
            s = s.to(device)
        for s in solvents:
            s = s.to(device)
        y = y.to(device).squeeze(-1)

        preds, _ = model(solutes, solvents)

        # MAE and its gradient
        loss = F.l1_loss(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)

    return total_loss / len(train_loader.dataset)


def predict(model, loader, device="cpu"):
    model.eval()
    pred_all = []
    y_all = []

    with t.no_grad():
        for solutes, solvents, y in loader:
            for s in solutes:
                s = s.to(device)
            for s in solvents:
                s = s.to(device)

            y = y.to(device).squeeze(-1)  # remove the last dimension
            preds, _ = model(solutes, solvents)
            pred_all.append(preds.detach().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())

    pred_all = np.concatenate(pred_all)
    y_all = np.concatenate(y_all)
    mae = np.mean(np.abs(pred_all - y_all))
    rmse = np.sqrt(np.mean((pred_all - y_all) ** 2))

    return mae, rmse, pred_all, y_all


if __name__ == "__main__":

    # fmt: off
    parser = argparse.ArgumentParser(description="r2s2-GAT model for solvation free energy prediction")
    parser.add_argument("--train-set", type=str, help="Path to train set in Torch DataLoader format (.pth)", required=True)
    parser.add_argument("--val-set", type=str, help="Path to validation set in Torch DataLoader format (.pth)", required=True)
    parser.add_argument("--test-set", type=str, default=None, help="Path to test set in Torch DataLoader format (.pth)")
    parser.add_argument("--atom-dim", type=int, default=30, help="Number of atom features")
    parser.add_argument("--bond-dim", type=int, default=6, help="Number of bond features")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--lambda-gmu", type=float, default=0.1, help="Residual factor in GMU")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--best-val", type=float, default=1e9, help="Best validation MAE (cutoff)")
    parser.add_argument("--save-model", type=str, help="Where to save the best model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    args = parser.parse_args()
    
    train_loader = t.load(args.train_set, weights_only=False, map_location=args.device)
    val_loader = t.load(args.val_set, weights_only=False, map_location=args.device)
    # fmt: on

    model = R2S2GATModel(
        atom_in=args.atom_dim,
        edge_in=args.bond_dim,
        n_heads=args.heads,
        lambda_gmu=args.lambda_gmu,
        device=args.device,
    ).to(args.device)
    optimizer = t.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_state = None

    for ep in range(1, args.epochs + 1):
        start_time = time.time()
        tr_loss = train(model, train_loader, optimizer, device=args.device)
        used_time = time.time() - start_time
        val_mae, val_rmse, _, _ = predict(model, val_loader, device=args.device)

        print(
            f"Epoch {ep} train_loss {tr_loss:.4f} val_mae {val_mae:.4f} val_rmse {val_rmse:.4f} time {used_time:.2f}s"
        )

        if val_mae < args.best_val:
            args.best_val = val_mae
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            t.save(model.state_dict(), os.path.join(args.save_model, "best_model.pt"))
            print(f"New best model saved! (Val Loss: {val_mae:.4f})")

    if args.save_model is not None:
        t.save(model.state_dict(), f"{args.save_model}/r2s2_model_epoch{ep}.pth")

    if args.test_set is not None:
        if best_state is not None:
            model.load_state_dict(best_state)

        test_loader = t.load(
            args.test_set, weights_only=False, map_location=args.device
        )
        test_mae, test_rmse, preds, ys = predict(model, test_loader, device=args.device)
        print(f"Test: MAE {test_mae:.4f} RMSE {test_rmse:.4f}")
