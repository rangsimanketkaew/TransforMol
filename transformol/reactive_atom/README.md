# GNN for Selecting Reactive Atoms

## Predict reactive atoms

### Implemented

- MPNN, GraphSAGE, GAT models
- Energy head (atomwise decomposition)
- Localization head (per-orbital logits --> softmax over atoms)
- Loss:
  ```
  Loss = Energy MSE + optional force loss + -beta * J_sur + orthogonality regularizer
  ```
- Training loop with increase of localization weight (β)
- Validation routine that calculates Pipek-Mezey (PM) via PySCF:
  - energy MAE
  - force MAE (if force provided)
  - surrogate J_sur v.s. true J_PM correlation
  - per-orbital per-atom AUC (ranking) using top-k ground-truth labeling
- Can save/load checkpoints
