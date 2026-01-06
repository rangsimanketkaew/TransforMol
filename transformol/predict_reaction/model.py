"""
Models for reaction prediction using MPNN and CVAE.

Updates:
    14.10.2025 Initial script [Rangsiman Ketkaew]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool


class MPNNLayer(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr="add")

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (N, node_dim)
            edge_index: Edge connectivity (2, E)
            edge_attr: Edge features (E, edge_dim)
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(msg_input)

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        out = self.update_mlp(update_input)
        return self.norm(x + out)


class MPNNEncoder(nn.Module):
    """
    MPNN Encoder for molecular graphs
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.conv_layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, edge_in_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch

        Returns:
            node_embeddings: Node-level embeddings (N, hidden_dim)
            graph_embedding: Graph-level embedding (batch_size, hidden_dim)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = (
            data.batch
            if hasattr(data, "batch")
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        h = self.node_embed(x)
        for conv in self.conv_layers:
            h = conv(h, edge_index, edge_attr)
            h = self.dropout(h)

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        graph_emb = self.graph_mlp(torch.cat([h_mean, h_max], dim=-1))

        return h, graph_emb


class ReactiveAtomPredictor(nn.Module):
    """
    Predicts which atoms are reactive (involved in bond breaking/forming).
    [Binary classification]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_embeddings):
        return torch.sigmoid(self.mlp(node_embeddings))


class CVAE(nn.Module):
    """
    Conditional variational autoencoder (CVAE) for generating TS and product structures

    The CVAE is conditioned on:
    1. Reactant graph representation
    2. Reactive atom predictions
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int = 64,
        node_out_dim: int = 25,  # Node feature dimension
        edge_out_dim: int = 6,  # Edge feature dimension
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.node_out_dim = node_out_dim
        self.edge_out_dim = edge_out_dim

        # Encoder: q(z | x, c) where x is target (TS/P), c is condition (reactant)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Condition + target
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: p(x | z, c)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),  # Latent + condition
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim),
        )

        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_out_dim),
        )

    def encode(self, target_emb, condition_emb):
        """
        Encode target and condition to latent distribution.

        Args:
            target_emb: Target graph embedding (TS or Product)
            condition_emb: Condition (Reactant) graph embedding

        Returns:
            mu, logvar: Parameters of latent distribution
        """
        h = self.encoder_mlp(torch.cat([target_emb, condition_emb], dim=-1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick:

        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition_emb, batch_num_atoms):
        h = self.decoder_mlp(torch.cat([z, condition_emb], dim=-1))

        if isinstance(batch_num_atoms, int):
            # Single graph
            h_expanded = h.repeat(batch_num_atoms, 1)
        else:
            # Batch of graphs
            h_expanded = h.repeat_interleave(batch_num_atoms, dim=0)

        node_features = self.node_decoder(h_expanded)

        return node_features, h_expanded

    def forward(self, target_emb, condition_emb, num_atoms):
        mu, logvar = self.encode(target_emb, condition_emb)
        z = self.reparameterize(mu, logvar)
        node_features, h_expanded = self.decode(z, condition_emb, num_atoms)

        return node_features, mu, logvar, h_expanded

    def sample(self, condition_emb, num_atoms, num_samples=1):
        device = condition_emb.device

        # Sample from prior N(0, I)
        z = torch.randn(num_samples, self.latent_dim, device=device)

        if condition_emb.dim() == 1:
            condition_emb = condition_emb.unsqueeze(0)

        condition_emb = condition_emb.repeat(num_samples, 1)
        node_features, h_expanded = self.decode(z, condition_emb, num_atoms)

        return node_features


class ReactionGenerativeModel(nn.Module):
    """
    Complete generative model for reaction prediction.

    Architecture:
    1. MPNN Encoder: Encodes reactant molecule
    2. Reactive Atom Predictor: Identifies reactive atoms
    3. CVAE: Generates TS and Product structures
    """

    def __init__(
        self,
        node_in_dim: int = 25,
        edge_in_dim: int = 6,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_mpnn_layers: int = 4,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.reactant_encoder = MPNNEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mpnn_layers,
        )

        self.ts_encoder = MPNNEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mpnn_layers,
        )

        self.product_encoder = MPNNEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mpnn_layers,
        )

        self.reactive_predictor = ReactiveAtomPredictor(hidden_dim)

        self.ts_cvae = CVAE(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            node_out_dim=node_in_dim,
            edge_out_dim=edge_in_dim,
        )

        self.product_cvae = CVAE(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            node_out_dim=node_in_dim,
            edge_out_dim=edge_in_dim,
        )

    def forward(self, reactant_data, ts_data=None, product_data=None, training=True):
        """
        Args:
            reactant_data: Reactant graph
            ts_data: TS graph (for training)
            product_data: Product graph (for training)
            training: Whether in training mode

        Returns:
            Dictionary with predictions and intermediate values
        """
        r_node_emb, r_graph_emb = self.reactant_encoder(reactant_data)
        reactive_scores = self.reactive_predictor(r_node_emb)

        if hasattr(reactant_data, "batch"):
            batch = reactant_data.batch
            batch_num_atoms = torch.bincount(batch)
        else:
            # Single graph
            batch_num_atoms = reactant_data.x.size(0)

        if training and ts_data is not None and product_data is not None:
            # Encode targets
            _, ts_graph_emb = self.ts_encoder(ts_data)
            _, p_graph_emb = self.product_encoder(product_data)

            ts_node_feat, ts_mu, ts_logvar, ts_h = self.ts_cvae(
                ts_graph_emb, r_graph_emb, batch_num_atoms
            )

            p_node_feat, p_mu, p_logvar, p_h = self.product_cvae(
                p_graph_emb, r_graph_emb, batch_num_atoms
            )

            return {
                "reactive_scores": reactive_scores,
                "ts_node_features": ts_node_feat,
                "ts_mu": ts_mu,
                "ts_logvar": ts_logvar,
                "p_node_features": p_node_feat,
                "p_mu": p_mu,
                "p_logvar": p_logvar,
                "r_node_emb": r_node_emb,
                "r_graph_emb": r_graph_emb,
            }
        else:
            # Inference mode - sample from prior
            return {
                "reactive_scores": reactive_scores,
                "r_node_emb": r_node_emb,
                "r_graph_emb": r_graph_emb,
            }

    def generate_reactions(self, reactant_data, num_samples=3):
        """
        Generate multiple plausible reactions.

        Args:
            reactant_data: Reactant graph
            num_samples: Number of reactions to generate

        Returns:
            List of generated TS and Product structures
        """
        self.eval()
        with torch.no_grad():
            r_node_emb, r_graph_emb = self.reactant_encoder(reactant_data)
            reactive_scores = self.reactive_predictor(r_node_emb)
            num_atoms = reactant_data.x.size(0)
            results = []
            for i in range(num_samples):
                ts_node_feat = self.ts_cvae.sample(
                    r_graph_emb, num_atoms, num_samples=1
                )

                p_node_feat = self.product_cvae.sample(
                    r_graph_emb, num_atoms, num_samples=1
                )

                results.append(
                    {
                        "ts_features": ts_node_feat.squeeze(0),
                        "product_features": p_node_feat.squeeze(0),
                        "reactive_scores": reactive_scores,
                    }
                )

            return results


def calculate_loss(model_output, targets, beta=0.001):
    """
    The loss function defined for this model:

    Loss = Reconstruction Loss + beta * KL Divergence + Reactive Atom Loss

    Args:
        model_output: Dictionary from model forward pass
        targets: Dictionary with target features and reactive labels
        beta: Weight for KL divergence term
    """
    # Reconstruction loss (MSE for node features)
    ts_recon_loss = F.mse_loss(model_output["ts_node_features"], targets["ts_features"])

    p_recon_loss = F.mse_loss(model_output["p_node_features"], targets["p_features"])

    # KL divergence loss
    def kl_divergence(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    ts_kl_loss = kl_divergence(model_output["ts_mu"], model_output["ts_logvar"])
    p_kl_loss = kl_divergence(model_output["p_mu"], model_output["p_logvar"])

    if "reactive_labels" in targets:
        reactive_loss = F.binary_cross_entropy(
            model_output["reactive_scores"].squeeze(-1),
            targets["reactive_labels"].float(),
        )
    else:
        reactive_loss = torch.tensor(0.0, device=ts_recon_loss.device)

    total_loss = (
        ts_recon_loss + p_recon_loss + beta * (ts_kl_loss + p_kl_loss) + reactive_loss
    )

    return {
        "total_loss": total_loss,
        "ts_recon_loss": ts_recon_loss,
        "p_recon_loss": p_recon_loss,
        "ts_kl_loss": ts_kl_loss,
        "p_kl_loss": p_kl_loss,
        "reactive_loss": reactive_loss,
    }
