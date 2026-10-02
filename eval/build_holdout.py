# --- Build the Held-Out Question Set ---
# 40 questions written AFTER agent v2 was finished, to check that v2's gain is
# not just a fit to the main suite. Agent v2 has not been changed since.
#
#   python -m eval.build_holdout        -> writes eval/holdout.jsonl
#
# What makes them different from the main suite:
#   rephrased        the same skills, asked in different words, mostly about
#                    tickers that appear nowhere in the main suite
#   new_combination  questions the main suite has no template for
#   trade            new orders, phrased differently, with a different confirmation
#   educational      new concepts
# Expected answers are computed the same way as in eval/build_questions.py.
import math
from collections import Counter

import eval.offline as offline   # must come before the agent imports

from agent.db import reset_database, get_recent_transactions
from agent.market import get_stock_price, get_historical_data, get_option_chain
from agent.analysis import get_portfolio_summary, get_sector_allocation, compare_stocks
from eval.build_questions import (
    EXACT, MONEY, PCT, CASH_AND_PRICE, PERIOD_WORDS,
    num, call, question, build_trade, cash, price, shares_owned, pct_change,
)

CONFIRM_MESSAGE = "Yes, do it."

SHARES_AND_PRICE = [["get_holding_by_ticker", "get_stock_price"],
                    ["get_holdings", "get_stock_price"],
                    ["get_portfolio_summary", "get_stock_price"]]


def build_rephrased():
    questions = []

    for ticker in ["PEP", "INTC"]:
        questions.append(question(
            "rephrased",
            f"I want to put all of my cash into {ticker}. What's the largest whole number of shares I can get?",
            CASH_AND_PRICE, [call("get_user_info"), call("get_stock_price", ticker=ticker)], 2,
            numbers=[num("whole_shares", math.floor(cash() / price(ticker)), EXACT)],
        ))

    for n, ticker in [(15, "MCD"), (30, "CSCO")]:
        questions.append(question(
            "rephrased",
            f"Suppose I picked up {n} shares of {ticker} today. What would my cash balance drop to? "
            "Don't actually buy anything.",
            CASH_AND_PRICE, [call("get_user_info"), call("get_stock_price", ticker=ticker)], 2,
            numbers=[num("cash_left", cash() - n * price(ticker), MONEY)],
        ))

    for held, target in [("VTI", "HD"), ("BND", "PFE")]:
        proceeds = shares_owned(held) * price(held)
        questions.append(question(
            "rephrased",
            f"Say I dumped my whole {held} position at today's price and rolled the money into {target}. "
            f"How much would the sale bring in, and how many full shares of {target} is that? Just the math, no orders.",
            SHARES_AND_PRICE,
            [call("get_holding_by_ticker", ticker=held), call("get_stock_price", ticker=held),
             call("get_stock_price", ticker=target)], 2,
            numbers=[num("proceeds", proceeds, MONEY),
                     num("whole_shares", math.floor(proceeds / price(target)), EXACT)],
        ))

    for a, b, period in [("ORCL", "CSCO", "6mo"), ("XOM", "BAC", "3mo")]:
        questions.append(question(
            "rephrased",
            f"Over the past {PERIOD_WORDS[period]}, did {a} or {b} do better? Tell me each one's percent move.",
            [["get_historical_data"]],
            [call("get_historical_data", ticker=a, period=period),
             call("get_historical_data", ticker=b, period=period)], 2,
            numbers=[num(f"{a}_pct_change", pct_change(a, period), PCT),
                     num(f"{b}_pct_change", pct_change(b, period), PCT)],
        ))

    for a, b in [("JNJ", "AMZN"), ("TSLA", "JPM")]:
        comparison = compare_stocks(a, b)
        questions.append(question(
            "rephrased",
            f"Give me the trailing P/E for {a} and for {b}, plus my share count in each.",
            [["compare_stocks", "get_holdings"], ["compare_stocks", "get_holding_by_ticker"],
             ["compare_stocks", "get_portfolio_summary"]],
            [call("compare_stocks", ticker_1=a, ticker_2=b), call("get_holdings")], 2,
            numbers=[num(f"{a}_pe", comparison["stock_1"]["trailing_pe"], MONEY),
                     num(f"{b}_pe", comparison["stock_2"]["trailing_pe"], MONEY),
                     num(f"{a}_shares", shares_owned(a), EXACT),
                     num(f"{b}_shares", shares_owned(b), EXACT)],
        ))

    transactions = get_recent_transactions()   # newest first
    for ticker in ["JPM", "AMZN"]:
        last_buy = next(t for t in transactions if t["ticker"] == ticker and t["action"] == "BUY")
        questions.append(question(
            "rephrased",
            f"What did my latest {ticker} buy cost per share, and where is it trading today?",
            [["get_recent_transactions", "get_stock_price"], ["get_recent_transactions", "get_portfolio_summary"]],
            [call("get_recent_transactions"), call("get_stock_price", ticker=ticker)], 2,
            numbers=[num("purchase_price", last_buy["price_per_share"], MONEY),
                     num("current_price", price(ticker), MONEY)],
        ))

    for ticker in ["V", "WMT"]:
        questions.append(question(
            "rephrased",
            f"Where is {ticker} trading, and is there any news on it?",
            [["get_stock_price", "get_stock_news"]],
            [call("get_stock_price", ticker=ticker), call("get_stock_news", ticker=ticker)], 2,
            numbers=[num("price", price(ticker), MONEY)],
        ))

    for ticker in ["GOOGL", "AMZN"]:
        chain = get_option_chain(ticker)
        questions.append(question(
            "rephrased",
            f"How many {ticker} shares am I holding, and what's the at-the-money call strike for the nearest expiry?",
            [["get_holding_by_ticker", "get_option_chain"], ["get_holdings", "get_option_chain"],
             ["get_portfolio_summary", "get_option_chain"]],
            [call("get_holding_by_ticker", ticker=ticker), call("get_option_chain", ticker=ticker)], 2,
            numbers=[num("shares", shares_owned(ticker), EXACT),
                     num("strike", chain["calls_near_atm"][0]["strike"], EXACT)],
        ))

    return questions


def build_new_combinations():
    summary = get_portfolio_summary()
    positions = {p["ticker"]: p for p in summary["positions"]}
    questions = []

    for ticker in ["JNJ", "VTI"]:
        questions.append(question(
            "new_combination",
            f"What's my unrealized gain or loss on {ticker}, in dollars?",
            [["get_portfolio_summary"]] + SHARES_AND_PRICE,
            [call("get_portfolio_summary")], 1,
            numbers=[num("gain_loss", positions[ticker]["gain_loss"], MONEY)],
        ))

    best = max(summary["positions"], key=lambda p: p["gain_loss_percent"])
    questions.append(question(
        "new_combination",
        "Which of my holdings is up the most in percentage terms since I bought it, and by what percent?",
        [["get_portfolio_summary"]], [call("get_portfolio_summary")], 1,
        numbers=[num("gain_loss_percent", best["gain_loss_percent"], PCT)],
        contains=[best["ticker"]],
    ))

    questions.append(question(
        "new_combination",
        "How much is my whole portfolio worth right now, and what would it be worth if AAPL fell 10% from today's price?",
        [["get_portfolio_summary"]], [call("get_portfolio_summary")], 1,
        numbers=[num("portfolio_value", summary["portfolio_value"], MONEY),
                 num("value_after_drop", summary["portfolio_value"] - 0.10 * positions["AAPL"]["current_value"], MONEY)],
    ))

    for ticker in ["PFE", "XOM"]:
        data = get_stock_price(ticker)
        questions.append(question(
            "new_combination",
            f"How far above its 52-week low is {ticker} trading, as a percentage of that low?",
            [["get_stock_price"], ["compare_stocks"]], [call("get_stock_price", ticker=ticker)], 1,
            numbers=[num("pct_above_low", (data["current_price"] - data["year_low"]) / data["year_low"] * 100, PCT)],
        ))

    for ticker, period in [("INTC", "1y"), ("MCD", "6mo")]:
        history = get_historical_data(ticker, period)["summary"]
        questions.append(question(
            "new_combination",
            f"What was {ticker}'s closing price at the start of the past {PERIOD_WORDS[period]}, what is it now, "
            "and what's the percent change between them?",
            [["get_historical_data"]], [call("get_historical_data", ticker=ticker, period=period)], 1,
            numbers=[num("first_close", history["first_close"], MONEY),
                     num("last_close", history["last_close"], MONEY),
                     num("pct_change", history["pct_change"], PCT)],
        ))

    recent = get_recent_transactions(days=30)
    questions.append(question(
        "new_combination",
        "How many trades have I made in the last 30 days, and how much money did they add up to?",
        [["get_recent_transactions"]], [call("get_recent_transactions", days=30)], 1,
        numbers=[num("trade_count", len(recent), EXACT),
                 num("total_amount", sum(t["total_amount"] for t in recent), MONEY)],
    ))

    top_sector = get_sector_allocation()["sector_allocation"][0]
    questions.append(question(
        "new_combination",
        "Which sector is my biggest, what percent of my invested money is in it, and what is ORCL trading at?",
        [["get_sector_allocation", "get_stock_price"]],
        [call("get_sector_allocation"), call("get_stock_price", ticker="ORCL")], 2,
        numbers=[num("sector_pct", top_sector["percentage"], PCT), num("price", price("ORCL"), MONEY)],
        contains=[top_sector["sector"]],
    ))

    total = 20 * price("NKE") + 20 * price("CSCO")
    questions.append(question(
        "new_combination",
        "If I wanted 20 shares of NKE and 20 shares of CSCO, what would the two cost together, and how much "
        "cash would I have left afterwards? I'm only asking, don't buy anything.",
        CASH_AND_PRICE,
        [call("get_user_info"), call("get_stock_price", ticker="NKE"), call("get_stock_price", ticker="CSCO")], 2,
        numbers=[num("total_cost", total, MONEY), num("cash_left", cash() - total, MONEY)],
    ))

    for ticker in ["KO", "DIS"]:
        put = get_option_chain(ticker)["puts_near_atm"][0]
        questions.append(question(
            "new_combination",
            f"For {ticker}'s nearest expiration, which put strike is closest to the current price, and what "
            "did that put last trade at?",
            [["get_option_chain"]], [call("get_option_chain", ticker=ticker)], 1,
            numbers=[num("strike", put["strike"], EXACT), num("last_price", put["last_price"], 0.05)],
        ))

    bac = get_stock_price("BAC")
    questions.append(question(
        "new_combination",
        "What are today's high and low for BAC, and what did it close at yesterday?",
        [["get_stock_price"]], [call("get_stock_price", ticker="BAC")], 1,
        numbers=[num("day_high", bac["day_high"], EXACT), num("day_low", bac["day_low"], EXACT),
                 num("previous_close", bac["previous_close"], EXACT)],
    ))

    for ticker, other in [("KO", "PEP"), ("DIS", "NFLX")]:
        questions.append(question(
            "new_combination",
            f"How has {ticker} done over the past year in percent, and what is its P/E ratio?",
            [["get_historical_data", "compare_stocks"]],
            [call("get_historical_data", ticker=ticker, period="1y"),
             call("compare_stocks", ticker_1=ticker, ticker_2=other)], 2,
            numbers=[num("pct_change", pct_change(ticker, "1y"), PCT),
                     num("trailing_pe", compare_stocks(ticker, other)["stock_1"]["trailing_pe"], MONEY)],
        ))

    return questions


def build_trades():
    cases = [
        ("Please purchase 2 shares of COST for me.", dict(action="BUY", ticker="COST", shares=2), None),
        ("I'd like to unload 4 of my JNJ shares.", dict(action="SELL", ticker="JNJ", shares=4), None),
        ("Get me 15 shares of PEP.", dict(action="BUY", ticker="PEP", shares=15), None),
        ("Sell 600 shares of VTI.", dict(action="SELL", ticker="VTI", shares=600),
         ["500", "maximum", "too large"]),
        ("Pick up 3 shares of AMC.", dict(action="BUY", ticker="AMC", shares=3),
         ["restricted"]),
        ("Sell 5 shares of META.", dict(action="SELL", ticker="META", shares=5),
         ["own", "hold", "cannot", "can't"]),
    ]
    return [build_trade(request, trade, keywords, CONFIRM_MESSAGE) for request, trade, keywords in cases]


def build_educational():
    questions = []
    for text, keywords in [("What is an ETF?", ["fund"]),
                           ("Why do people say diversification lowers risk?", ["risk"])]:
        q = question("educational", text, [], [], 0, contains=keywords)
        q["expect_no_tools"] = True
        questions.append(q)
    return questions


def build_all():
    reset_database()
    questions = build_rephrased() + build_new_combinations() + build_trades() + build_educational()
    for index, q in enumerate(questions, start=1):
        q["id"] = f"holdout-{index:02d}"
        q["split"] = "holdout"
    return questions


if __name__ == "__main__":
    questions = build_all()
    assert len(questions) == 40, f"Expected 40 questions, built {len(questions)}"

    offline.write_jsonl(offline.HOLDOUT_PATH, questions)

    print(f"Wrote {len(questions)} questions to {offline.HOLDOUT_PATH}")
    for category, count in Counter(q["category"] for q in questions).items():
        print(f"  {category:<18} {count}")
    print(f"Multi-step: {sum(1 for q in questions if q['min_tool_calls'] >= 2 or len(q['turns']) > 1)}")
