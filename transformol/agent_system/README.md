## TransforMol LLM Agent System

An agent built with LangChain that routes chemistry queries from the user to one of four ML tools:

- Solvation Gibbs free energy prediction (R2S2-GAT)
- Reaction prediction (MPNN + CVAE)
- Reactive atom prediction (GNN + PM localization)
- Solute structure prediction (MoleculeMLP)

Usage
-----

```python
from transformol.agent_system import build_agent, run_agent, TransforMolAgentConfig

config = TransforMolAgentConfig(llm_provider="google")
result = run_agent("Predict solvation free energy of CC in water", config)
print(result)
```
