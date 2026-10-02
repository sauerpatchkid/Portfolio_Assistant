# --- Record a Market Data Snapshot ---
# Calls the live yfinance-backed tool functions once for a fixed set of tickers
# and saves their outputs. Evals and benchmarks then read this file instead of
# calling yfinance, so every run sees the same market data.
#
#   python -m scripts.record_snapshot
#
# After re-recording, rebuild the questions: python -m eval.build_questions
import argparse
import json
import time
from datetime import datetime, timezone

import yfinance as yf

from agent import market
from agent.db import MOCK_HOLDINGS

HELD_TICKERS = [ticker for ticker, _, _ in MOCK_HOLDINGS]
OTHER_TICKERS = ["META", "NFLX", "AMD", "KO", "DIS", "COST", "V", "WMT"]


def record(tickers, pause_s=0.5):
    market.use_snapshot(None)   # make sure we're recording live data

    snapshot = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"yfinance {yf.__version__}",
        "get_stock_price": {},
        "get_option_chain": {},
        "get_stock_news": {},
        "get_historical_data": {},
        "info": {},
    }
    problems = []

    def keep(section, key, result):
        if isinstance(result, dict) and "error" in result:
            problems.append(f"{section} {key}: {result['error']}")
            return
        snapshot[section][key] = result

    for ticker in tickers:
        print(f"Recording {ticker}...")
        keep("get_stock_price", ticker, market.get_stock_price(ticker))
        keep("get_option_chain", ticker, market.get_option_chain(ticker))
        keep("get_stock_news", ticker, market.get_stock_news(ticker))
        keep("info", ticker, market.get_company_info(ticker))
        for period in sorted(market.VALID_HISTORICAL_PERIODS):
            keep("get_historical_data", f"{ticker}|{period}", market.get_historical_data(ticker, period))
            time.sleep(pause_s)

    return snapshot, problems


SECTIONS = ["get_stock_price", "get_option_chain", "get_stock_news", "get_historical_data", "info"]


def add_tickers(path, tickers):
    """Record extra tickers into an existing snapshot without touching what is already there."""
    with open(path, encoding="utf-8") as f:
        snapshot = json.load(f)

    new = [t for t in tickers if t not in snapshot["get_stock_price"]]
    addition, problems = record(new)
    for section in SECTIONS:
        snapshot[section].update(addition[section])
    snapshot.setdefault("added", []).append({"recorded_at": addition["recorded_at"], "tickers": new})
    return snapshot, problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/market_snapshot.json")
    parser.add_argument("--add", nargs="+", metavar="TICKER",
                        help="Add these tickers to the existing snapshot instead of re-recording it")
    args = parser.parse_args()

    if args.add:
        snapshot, problems = add_tickers(args.out, args.add)
    else:
        snapshot, problems = record(HELD_TICKERS + OTHER_TICKERS)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=1)

    print(f"\nSaved {args.out}")
    for section in SECTIONS:
        print(f"  {section}: {len(snapshot[section])} entries")
    if problems:
        print(f"\n{len(problems)} lookups failed and were left out:")
        for p in problems:
            print(f"  - {p}")
