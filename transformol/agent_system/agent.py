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


class TransforMolAgentWrapper:
    def __init__(self, graph, tools):
        self.graph = graph
        self.tools = tools

    def invoke(self, inputs):
        query = inputs.get("input", "")
        res = self.graph.invoke({"messages": [{"role": "user", "content": query}]})
        output_content = ""
        if "messages" in res and res["messages"]:
            content = res["messages"][-1].content
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                output_content = "".join(text_parts)
            else:
                output_content = str(content)
        return {"output": output_content}


def build_agent(config):
    """Build and return a LangChain ReAct AgentExecutor wrapper.

    Parameters
    ----------
    config : TransforMolAgentConfig

    Returns
    -------
    TransforMolAgentWrapper
    """

    from langchain.agents import create_agent
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

    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_PROMPT,
    )

    return TransforMolAgentWrapper(graph, tools)


from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage

class MockChatModel(SimpleChatModel):
    """This is just a mock chat model for test"""
    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        user_query = ""
        has_tool_run = False
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type == "human":
                user_query = str(msg.content)
            elif msg_type in ("ai", "tool"):
                has_tool_run = True
            elif hasattr(msg, "content") and "predict_" in str(msg.content):
                has_tool_run = True

        if "DMSO" in user_query or "geometry" in user_query:
            if has_tool_run:
                return "The solute structure prediction in implicit solvent for ethanol (SMILES: 'CCO') in DMSO returned " + \
                    "a demo mode message because no model checkpoint was provided. Check `solv_strc_checkpoint` in the configuration."
            else:
                return "Thought: I need to predict the solute structure of ethanol in DMSO. " + \
                    "I should use the predict_solute_structure tool.\nAction: predict_solute_structure" + \
                    "\nAction Input: {\"smiles\": \"CCO\", \"solvent\": \"DMSO\"}"

        elif "reaction" in user_query:
            if has_tool_run:
                return "The reaction prediction for ethanol (SMILES: 'CCO') returned a demo mode message " + \
                    "because no model checkpoint was provided. Check `reaction_checkpoint` in the configuration."
            else:
                return "Thought: I need to predict the reaction outcome for ethanol (SMILES: CCO). " + \
                    "I should use the predict_reaction tool.\nAction: predict_reaction" + \
                    "\nAction Input: {\"smiles\": \"CCO\"}"

        elif "cyclohexane" in user_query or "reactive atoms" in user_query:
            if has_tool_run:
                return "The reactive atom prediction for cyclohexane (SMILES: 'C1CCCCC1') returned a demo mode message " + \
                    "because no model checkpoint was provided. Check `reactive_atom_checkpoint` in the configuration."
            else:
                return "Thought: I need to predict the reactive atoms for cyclohexane. " + \
                    "I should use the predict_reactive_atoms tool.\nAction: predict_reactive_atoms" + \
                    "\nAction Input: {\"smiles\": \"C1CCCCC1\"}"

        elif "ethane" in user_query or "solvation" in user_query or "CC" in user_query:
            if has_tool_run:
                return "The solvation Gibbs free energy prediction for ethane (SMILES: 'CC') in water returned " + \
                    "a demo mode message because no model checkpoint was provided. Check `solv_deltag_checkpoint` in the configuration."
            else:
                return "Thought: I need to predict the solvation Gibbs free energy of CC (ethane) in water. " + \
                    "I should use the predict_solvation_free_energy tool.\nAction: predict_solvation_free_energy\nAction " + \
                    "Input: {\"solute_smiles\": \"CC\", \"solvent\": \"water\"}"

        return "I am TransforMol agent. How can I help you today? :)"

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "mock"


def _build_llm(config):
    import os
    if os.getenv("MOCK_LLM") == "1":
        return MockChatModel()

    if config.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.llm_model, temperature=config.temperature, max_tokens=4096)
    elif config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.llm_model, temperature=config.temperature)
    elif config.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.llm_model, temperature=config.temperature)
    else:
        raise ValueError(
            f"Unknown llm_provider '{config.llm_provider}'. "
            "Choose 'anthropic', 'openai', or 'google'."
        )


def _offline_react_prompt():
    """Minimal ReAct prompt for offline environments"""

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
    """Build the agent and run a single query. Returns the final answer string"""

    result = build_agent(config).invoke({"input": query})
    return result.get("output", str(result))
