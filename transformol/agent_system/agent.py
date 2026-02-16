"""
TransforMol LLM Agents

Updates:
    13.02.2026  Initial implementation [Rangsiman Ketkaew]
"""


AGENT_PROMPT = """You are **TransforMol**, an expert computational chemistry assistant!
You have access to 4 ML tools:

1. predict_solvation_free_energy  - solvation Gibbs free energy (kcal/mol) via R2S2-GAT
2. predict_reaction               - transition state and product prediction via MPNN + CVAE
3. predict_reactive_atoms         - reactive atom ranking via GNN + Pipek-Mezey
4. predict_solute_structure       - solute geometry in implicit solvent via MLP

Some guidelines
---------------
- Use the correct tool for each task; pass inputs as JSON as described
- Solvent names (e.g. "water", "DMSO") and SMILES are both accepted
- Explain results in plain language
- If a tool returns a "demo mode" message, inform the user that a trained
  checkpoint is needed and explain how to set it via config
"""


def build_agent(config):
    """Build and return a LangChain ReAct AgentExecutor.

    Parameters
    ----------
    config : TransforMolAgentConfig

    Returns
    -------
    langchain.agents.AgentExecutor
    """
    try:
        from langchain.agents import AgentExecutor, create_react_agent
    except ImportError as exc:
        raise ImportError(
            "LangChain is not installed. Run:\n"
            "  pip install langchain langchain-core langchain-anthropic "
            "langchain-openai langchain-google-genai\n"
            f"Details: {exc}"
        ) from exc

    from .tools.solv_deltag_tool import build_solv_deltag_tool
    from .tools.reaction_tool import build_reaction_tool
    from .tools.reactive_atom_tool import build_reactive_atom_tool
    from .tools.solv_strc_tool import build_solv_strc_tool

    tools = [
        build_solv_deltag_tool(config),
        build_reaction_tool(config),
        build_reactive_atom_tool(config),
        build_solv_strc_tool(config),
    ]

    llm = _build_llm(config)

    try:
        from langchain import hub
        prompt = hub.pull("hwchase17/react")
    except Exception:
        prompt = _offline_react_prompt()

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=config.verbose,
        max_iterations=config.max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


def _build_llm(config):
    if config.llm_provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError("Run: pip install langchain-anthropic") from exc
        return ChatAnthropic(model=config.llm_model, temperature=config.temperature, max_tokens=4096)

    elif config.llm_provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("Run: pip install langchain-openai") from exc
        return ChatOpenAI(model=config.llm_model, temperature=config.temperature)

    elif config.llm_provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError("Run: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(model=config.llm_model, temperature=config.temperature)

    else:
        raise ValueError(
            f"Unknown llm_provider '{config.llm_provider}'. "
            "Choose 'anthropic', 'openai', or 'google'."
        )


def _offline_react_prompt():
    """Minimal ReAct prompt for offline / Hub-unavailable environments."""
    from langchain_core.prompts import PromptTemplate
    template = (
        AGENT_PROMPT + "\n\n"
        "You have access to the following tools:\n\n{tools}\n\n"
        "Use the format:\n"
        "Question: {input}\n"
        "Thought: ...\n"
        "Action: one of [{tool_names}]\n"
        "Action Input: ...\n"
        "Observation: ...\n"
        "Thought: I now know the final answer\n"
        "Final Answer: ...\n\n"
        "Begin!\n\nQuestion: {input}\nThought:{agent_scratchpad}"
    )
    return PromptTemplate.from_template(template)


def run_agent(query, config):
    """Build the agent and run a single query. Returns the final answer string."""
    result = build_agent(config).invoke({"input": query})
    return result.get("output", str(result))
