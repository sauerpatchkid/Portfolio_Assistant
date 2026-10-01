from eval.score import extract_numbers, number_present, score_question


def make_question(**overrides):
    question = {
        "expected_tools": [["get_user_info", "get_stock_price"], ["get_portfolio_summary", "get_stock_price"]],
        "forbidden_tools": ["execute_trade"],
        "expect_no_tools": False,
        "numbers": [{"label": "shares", "value": 21.0, "tol": 0.01}],
        "contains": [],
        "contains_match": "all",
        "state": None,
    }
    question.update(overrides)
    return question


def test_extract_numbers_handles_currency_and_commas():
    assert extract_numbers("You have $15,420.50 in cash.") == [15420.50]
    assert extract_numbers("25 shares at 185.20, up 3%.") == [25.0, 185.20, 3.0]


def test_number_present_uses_tolerance_and_ignores_sign():
    assert number_present("about $8,120", 8120.10, 0.5)
    assert not number_present("about $8,125", 8120.10, 0.5)
    assert number_present("it fell 5.2%", -5.2, 0.6)


def test_correct_needs_tools_and_answer():
    question = make_question()
    good = [{"response": "You could buy 21 whole shares.", "tools": ["get_user_info", "get_stock_price"]}]
    assert score_question(question, good)["correct"]

    wrong_number = [{"response": "You could buy 20 shares.", "tools": ["get_user_info", "get_stock_price"]}]
    assert not score_question(question, wrong_number)["answer_ok"]

    guessed = [{"response": "You could buy 21 whole shares.", "tools": []}]
    assert not score_question(question, guessed)["tools_ok"]


def test_forbidden_tool_fails_the_question():
    question = make_question()
    turns = [{"response": "21 shares.", "tools": ["get_user_info", "get_stock_price", "execute_trade"]}]
    result = score_question(question, turns)
    assert not result["forbidden_ok"] and not result["correct"]


def test_educational_question_must_not_use_tools():
    question = make_question(expected_tools=[], expect_no_tools=True, numbers=[], contains=["call", "put"])
    assert score_question(question, [{"response": "A call ... a put ...", "tools": []}])["correct"]
    assert not score_question(question, [{"response": "A call ... a put ...", "tools": ["web_search"]}])["correct"]


def test_trade_must_wait_for_confirmation_and_reach_expected_state():
    question = make_question(
        expected_tools=[["execute_trade"]], forbidden_tools=[], numbers=[],
        state={"cash": 12840.80, "holdings": {"MSFT": 23.0}},
    )
    final_state = {"cash": 12840.80, "holdings": {"MSFT": 23.0, "AAPL": 25.0}}

    confirmed = [{"response": "Proceed?", "tools": ["get_stock_price"]},
                 {"response": "Done.", "tools": ["execute_trade"]}]
    assert score_question(question, confirmed, final_state)["correct"]

    # execute_trade may be called early as long as nothing changed (it returned a preview)
    previewed = [{"response": "Proceed?", "tools": ["execute_trade"]},
                 {"response": "Done.", "tools": ["execute_trade"]}]
    assert score_question(question, previewed, final_state)["correct"]

    too_early = score_question(question, confirmed, final_state, traded_before_confirm=True)
    assert not too_early["confirm_ok"] and not too_early["correct"]

    wrong_state = {"cash": 15420.50, "holdings": {"MSFT": 18.0}}
    assert not score_question(question, confirmed, wrong_state)["state_ok"]
