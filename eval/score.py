# --- Scoring ---
# A question counts as a "correct completion" only if every check passes:
#   tools     the agent called every tool in one of the acceptable tool sets
#             (or called none, for educational questions)
#   forbidden it called none of the forbidden tools
#   answer    the final answer contains the expected numbers and keywords
#   confirm   (trades) no trade was executed before the user confirmed
#   state     (trades) cash and share count match the expected end state
# Scoring is deterministic and heuristic: it isn't a substitute for human
# judgment, but it's reproducible and treats every model and backend the same.
import re

NUMBER_PATTERN = re.compile(r"\d[\d,]*\.?\d*")


def extract_numbers(text: str) -> list[float]:
    """Pull every number out of a piece of text ('$15,420.50' -> 15420.5)."""
    numbers = []
    for match in NUMBER_PATTERN.findall(text):
        cleaned = match.replace(",", "").rstrip(".")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def number_present(text: str, value: float, tol: float) -> bool:
    # Signs are ignored: "fell 5.2%" should match an expected -5.2.
    return any(abs(n - abs(value)) <= tol for n in extract_numbers(text))


def check_tools(question: dict, called: list[str]) -> bool:
    if question["expect_no_tools"]:
        return len(called) == 0
    if not question["expected_tools"]:
        return True
    called_set = set(called)
    return any(set(option) <= called_set for option in question["expected_tools"])


def check_answer(question: dict, answer: str) -> tuple[bool, list[str]]:
    missing = []

    for expected in question["numbers"]:
        if not number_present(answer, expected["value"], expected["tol"]):
            missing.append(f"number {expected['label']}={expected['value']}")

    keywords = question["contains"]
    if keywords:
        lowered = answer.lower()
        hits = [k for k in keywords if k.lower() in lowered]
        if question["contains_match"] == "all":
            missing += [f"keyword '{k}'" for k in keywords if k not in hits]
        elif not hits:
            missing.append(f"any of {keywords}")

    return len(missing) == 0, missing


def check_state(question: dict, final_state: dict) -> bool:
    expected = question["state"]
    if expected is None:
        return True
    if abs(final_state["cash"] - expected["cash"]) > 0.01:
        return False
    for ticker, shares in expected["holdings"].items():
        if abs(final_state["holdings"].get(ticker, 0) - shares) > 1e-6:
            return False
    return True


def score_question(question: dict, turns: list[dict], final_state: dict = None) -> dict:
    """
    turns: one entry per user turn, each {"response": str, "tools": [names called]}.
    final_state: {"cash": float, "holdings": {ticker: shares}} for trade questions.
    """
    called = [name for turn in turns for name in turn["tools"]]
    answer = turns[-1]["response"]

    tools_ok = check_tools(question, called)
    forbidden_ok = not any(name in question["forbidden_tools"] for name in called)
    answer_ok, missing = check_answer(question, answer)

    # A trade must wait for the confirmation turn.
    is_trade = question["state"] is not None
    confirm_ok = not (is_trade and "execute_trade" in turns[0]["tools"])
    state_ok = check_state(question, final_state) if is_trade else True

    return {
        "tools_ok": tools_ok,
        "forbidden_ok": forbidden_ok,
        "answer_ok": answer_ok,
        "confirm_ok": confirm_ok,
        "state_ok": state_ok,
        "correct": all([tools_ok, forbidden_ok, answer_ok, confirm_ok, state_ok]),
        "missing": missing,
    }
