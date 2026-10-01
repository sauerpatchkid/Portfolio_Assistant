# --- Offline Eval Environment ---
# Import this module BEFORE anything from `agent`. It points the agent at a
# separate eval database and at the recorded market snapshot, so eval runs are
# repeatable and never touch the demo database or the live yfinance API.
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

os.environ.setdefault("PORTFOLIO_DB", str(RESULTS_DIR / "eval_portfolio.db"))
os.environ.setdefault("MARKET_SNAPSHOT", str(ROOT / "data" / "market_snapshot.json"))

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
TRACE_PATH = ROOT / "bench" / "trace.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_csv_row(path, row: dict):
    """Add one row to a summary CSV, writing the header if the file is new."""
    is_new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
