# TransforMol

*TransforMol* is an LLM quantum-ML chemistry assistant that routes natural-language queries to specialized graph transformer and MLP models. It is built on LangChain and uses 4 quantum chemistry-based ML prediction tools as a ReAct agent.

## Quantum Chemistry ML Tasks

| # | Task | Model | Input | Output |
|---|------|-------|-------|--------|
| 1 | **Solvation Gibbs free energy** ([code](transformol/solv_deltaG)) | R2S2-GAT | solute SMILES + solvent | ΔG (kcal/mol) |
| 2 | **Reaction prediction** ([code](transformol/predict_reaction)) | MPNN + CVAE | reactant SMILES | TS & product feature norms, reactive atom ranking |
| 3 | **Reactive atom prediction** ([code](transformol/reactive_atom)) | GNN + Pipek-Mezey | SMILES (+ optional XYZ) | Per-atom reactivity scores and ranking |
| 4 | **Solute structure in implicit solvent** ([code](transformol/solv_strc)) | MoleculeMLP | SMILES or XYZ + solvent | Per-atom displacement vectors (Å) and RMSD |

> *R2S2-GAT is a specialized GAT architecture built with PyTorch Geometric, designed to couple the interaction between solute and solvents.*

## Install

```bash
# fork to your own GitHub first (optional)
git clone https://github.com/rangsimanketkaew/TransforMol.git
cd TransforMol
pip install langchain langchain-core
# Install pretrained LLM provider (at least one)
pip install langchain-anthropic # Anthropic Claude
pip install langchain-openai # OpenAI GPT
pip install langchain-google-genai # Google Gemini
# Install dependencies for core quantum chemistry ML models
pip install torch torch-geometric rdkit numpy pandas h5py scikit-learn
```

## LLM Setup

Set the API key for your chosen provider as an environment variable

```bash
# Anthropic (default - claude-sonnet-4-5)
export ANTHROPIC_API_KEY="XXX"

# OpenAI (gpt-4o)
export OPENAI_API_KEY="XXX"

# Google (gemini-2.0-flash)
export GOOGLE_API_KEY="XXX"
```

## Example usage

```python
from transformol.agent_system import build_agent, run_agent, TransforMolAgentConfig

# Choose provider between "anthropic" | "openai" | "google"
config = TransforMolAgentConfig(llm_provider="anthropic")

# Single-call helper
result = run_agent("What is the solvation free energy of ethanol in water?", config)
print(result)
```

With pre-trained model checkpoints:

```python
config = TransforMolAgentConfig(
    llm_provider="openai",
    solv_deltag_checkpoint="path/to/r2s2_model.pt",
    reactive_atom_checkpoint="path/to/reactive_atom.pt",
    reaction_checkpoint="path/to/reaction_model.pt",
    solv_strc_checkpoint="path/to/solv_strc.pt",
    solv_strc_metadata="path/to/metadata.json",
    device="cuda",
)

agent = build_agent(config)
result = agent.invoke({"input": "Rank the reactive atoms of cyclohexane"})
print(result["output"])
```

> **Note**: Without checkpoints the tools run in *demo mode* and return informative messages instead of real predictions.

## Running the Demo

With Python CLI

```bash
python -c "
from transformol.agent_system import run_agent, TransforMolAgentConfig
config = TransforMolAgentConfig(llm_provider='google')
print(run_agent('Predict the solvation free energy of CC in water', config))
"
```

## Developer

[Rangsiman Ketkaew](https://rangsimanketkaew.github.io)
ETH Zurich, Switzerland
rangsiman.ketkaew@phys.chem.ethz.ch

## License

See [MIT License](LICENSE)
