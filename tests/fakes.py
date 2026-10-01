# --- Stand-in Models for Testing Without a GPU ---
from agent.backends import ChatTurn


class OracleBackend:
    """
    A stand-in model that behaves perfectly: it makes each question's oracle
    tool calls, then answers with the expected numbers and keywords. Running
    the eval harness against it checks the harness itself, with no GPU.
    """

    def __init__(self, questions):
        self.by_first_message = {q["turns"][0]: q for q in questions}
        self.calls = 0

    async def chat(self, messages, tools=None, max_tokens=512) -> ChatTurn:
        self.calls += 1
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        question = self.by_first_message[user_messages[0]]
        oracle_turn = question["oracle"][len(user_messages) - 1]

        # Tool results since the last user message tell us how far along we are.
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        done = sum(1 for m in messages[last_user:] if m["role"] == "tool")

        if done < len(oracle_turn["calls"]):
            call = oracle_turn["calls"][done]
            return ChatTurn(
                content="",
                tool_calls=[{"id": f"call_{self.calls}", "name": call["name"], "arguments": call["arguments"]}],
                prompt_tokens=100, completion_tokens=20,
            )

        if oracle_turn["reply"] is not None:
            answer = oracle_turn["reply"]
        else:
            parts = [f"{n['label']}: {n['value']}" for n in question["numbers"]] + question["contains"]
            answer = "Answer. " + ", ".join(parts)
        return ChatTurn(content=answer, prompt_tokens=100, completion_tokens=20)


class SilentBackend:
    """A stand-in model that never calls tools and never says anything useful."""

    async def chat(self, messages, tools=None, max_tokens=512) -> ChatTurn:
        return ChatTurn(content="I don't know.", prompt_tokens=100, completion_tokens=5)
