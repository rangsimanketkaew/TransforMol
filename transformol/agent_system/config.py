"""
Configuration
Set llm_provider and provide the matching API key via env var. 

Updates:
    13.02.2026  Initial implementation [Rangsiman Ketkaew]
"""

from dataclasses import dataclass


@dataclass
class TransforMolAgentConfig:
    """Configuration variables for LLM agent!"""

    # LLM
    llm_provider: str = "anthropic" # "anthropic" | "openai" | "google"
    llm_model: str | None = None # None = provider default
    temperature: float = 0.0 # 0.0 = deterministic, >0 = stochastic
    max_iterations: int = 10 # max agent loop iterations

    # Solvation Gibbs free energy
    solv_deltag_checkpoint: str | None = None # Path to R2S2-GAT checkpoint
    solv_deltag_atom_dim: int = 30 # Atom feature dimension (must match training)
    solv_deltag_bond_dim: int = 6 # Bond feature dimension (must match training)

    # Reaction prediction
    reaction_checkpoint: str | None = None # Path to MPNN + CVAE checkpoint
    reaction_node_dim: int = 25 # Node feature dimension (must match training)
    reaction_edge_dim: int = 6 # Bond feature dimension (must match training)
    reaction_hidden_dim: int = 128 # Hidden dimension (must match training)
    reaction_latent_dim: int = 64 # Latent dimension (must match training)
    reaction_num_samples: int = 3 # Number of reaction hypotheses to sample

    # Reactive atom prediction
    reactive_atom_checkpoint: str | None = None # Path to GNN + PM checkpoint
    reactive_atom_model_type: str = "mpnn" # "mpnn" | "sage" | "gat"
    reactive_atom_hidden_dim: int = 128 # Hidden dimension (must match training)
    reactive_atom_n_orb: int = 8 # Number of orbital features (must match training)
    reactive_atom_mode: str = "sum" # "sum" | "max" | "mean"

    # Solute structure
    solv_strc_checkpoint: str | None = None # Path to MoleculeMLP checkpoint
    solv_strc_metadata: str | None = None # Path to metadata.json from training

    # General
    device: str = "cpu" # "cpu" | "cuda"
    verbose: bool = True # Print agent reasoning steps

    def __post_init__(self):
        if self.llm_model is None:
            defaults = {
                "anthropic": "claude-sonnet-4-5",
                "openai": "gpt-4o",
                "google": "gemini-2.0-flash",
            }
            if self.llm_provider not in defaults:
                raise ValueError(
                    f"Unknown llm_provider '{self.llm_provider}'. "
                    "Choose 'anthropic', 'openai', or 'google'."
                )
            self.llm_model = defaults[self.llm_provider]
