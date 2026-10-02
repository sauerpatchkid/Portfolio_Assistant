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
HOLDOUT_PATH = ROOT / "eval" / "holdout.jsonl"
TRACE_PATH = ROOT / "bench" / "trace.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_csv_row(path, row: dict):
    """
    Add one row to a summary CSV. The file is rewritten with the union of the
    existing columns and the row's columns, so a row whose columns differ from
    the header (an older file, a newer script) still lines up.
    """
    rows = []
    columns = []
    if Path(path).exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows = list(reader)

    columns += [c for c in row if c not in columns]
    rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
