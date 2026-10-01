# --- Prompt Versions ---
# v1 is the system prompt and tool schemas from the original course notebook.
# Later versions are added here after reading the v1 failures in the eval suite,
# so every run can name exactly which prompt it used (--prompt v1).
import copy

from agent.tools import TOOLS

BASE_SYSTEM_PROMPT = """You are a professional financial portfolio assistant. You help users manage their investment portfolio, check market data, execute trades, and learn about financial concepts.

RULES:
1. Always use tools to look up real data — never make up prices, balances, or holdings.
2. For trade requests, FIRST tell the user what you will do (ticker, shares, estimated cost) and ask for confirmation. Only call execute_trade after they confirm.
3. For educational questions about financial concepts, answer directly without tools.
4. Be concise but thorough. Format numbers as currency where appropriate.
5. If a tool returns an error, explain the issue clearly.
"""

SYSTEM_PROMPTS = {
    "v1": BASE_SYSTEM_PROMPT,
}

# Per-version replacements for tool descriptions: {version: {tool_name: description}}
TOOL_DESCRIPTION_OVERRIDES = {
    "v1": {},
}


def get_prompt_config(version: str = "v1"):
    """Return (system_prompt, tools) for a prompt version."""
    if version not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt version '{version}'. Known: {sorted(SYSTEM_PROMPTS)}")

    tools = copy.deepcopy(TOOLS)
    overrides = TOOL_DESCRIPTION_OVERRIDES.get(version, {})
    for tool in tools:
        name = tool["function"]["name"]
        if name in overrides:
            tool["function"]["description"] = overrides[name]

    return SYSTEM_PROMPTS[version], tools
