# Portfolio Assistant

A virtual assistant for a personal investment portfolio, built on open-weights
Qwen3 models (0.6B and 8B). It answers questions about an account, looks up
market data, analyzes the portfolio, explains financial concepts, and places
trades behind guardrails. Every number in an answer comes from a tool result,
not from the model's memory.

```
You:        How many whole shares of META could I buy with my available cash
            at its current price?
Assistant:  [calls get_user_info, then get_stock_price("META")]
            With your available cash of $15,420.50 and the current price of META
            at $730.04, you could buy approximately 21 whole shares.
```

## What it can do

| Area | Examples |
|---|---|
| Account | Cash balance, holdings, cost basis, recent trades |
| Market data | Prices, 52-week range, option chains, news, price history |
| Portfolio analysis | Total value and P&L, sector allocation, side-by-side stock comparison |
| Education | Explains concepts such as covered calls or P/E ratio, without calling tools |
| Trading | Buys and sells, market or limit orders, only after the user confirms |

## How it works

**Data layer.** A SQLite database holds the user, holdings and transaction
history, seeded with a mock portfolio of ten positions. Market data comes from
yfinance.

**Tools.** The model reaches the data through 13 tools, each described to it as
a JSON schema:

| Group | Tools |
|---|---|
| Account | `get_user_info`, `get_holdings`, `get_holding_by_ticker`, `get_recent_transactions` |
| Market data | `get_stock_price`, `get_option_chain`, `get_stock_news`, `get_historical_data` |
| Analysis | `get_portfolio_summary`, `get_sector_allocation`, `compare_stocks` |
| Web | `web_search` |
| Trading | `execute_trade` |

**Agent loop.** The model reads the question and either answers or asks for
tools. The tool results are appended to the conversation and the model is
called again, for up to five rounds (`agent/agent.py`).

**Trade guardrails.** The prompt tells the model to describe a trade and wait
for confirmation before executing it. The code enforces the rest regardless of
what the model does: at most 500 shares per order, no restricted tickers, no
buying beyond the cash balance, no selling more than is held.

**Two model sizes.** Qwen3-0.6B and Qwen3-8B run the same agent, so the effect
of model size on tool use and answer quality can be compared directly. The 8B
model is 4-bit quantized so it fits on a 16 GB GPU.

## Evaluation

A 100-question suite, 84 of them multi-step (two or more tool calls, or a
two-turn trade that must wait for confirmation):

| Kind | Count | Example |
|---|---|---|
| Single lookup | 10 | "What is the current price of NVDA?" |
| Educational (must use no tools) | 6 | "What is dollar-cost averaging?" |
| Multi-tool | 72 | "How many whole shares of META could I buy with my available cash at its current price?" |
| Trades (request, then confirm) | 12 | "Buy 5 shares of MSFT." / "Buy 1000 shares of NVDA." (must be refused) |

A question is a **correct completion** only if the agent called the right tools,
called no forbidden tool, gave an answer containing the expected numbers, and
(for trades) waited for confirmation and left the database in the expected
state. Scoring is deterministic; see `eval/score.py`.

Three choices keep the suite repeatable:

- **Recorded market data.** Live prices change every minute, so evals read a
  snapshot of the tool outputs for 18 tickers (`data/market_snapshot.json`).
- **Computed answers.** `eval/build_questions.py` works out every expected
  number by calling the same tool functions the agent uses.
- **Dev and test halves.** Questions alternate between two halves inside every
  category, so prompt changes can be tuned on one and checked on the other.

## Latency

The original version of this project called `model.generate()` inside a
notebook, one request at a time. Two things make the assistant faster.

**Serving the model behind an API.** The agent talks to an OpenAI-compatible
endpoint, and the model runs in a separate server (vLLM). The server batches
concurrent requests on the GPU, so the eval suite can run many questions at
once instead of one after another. SGLang exposes the same API and is included
as an alternative backend; switching is a `--base-url` change.

**Reusing the KV cache.** Every request starts with the same system prompt and
13 tool schemas, and each step of a tool-calling turn repeats everything before
it. A server that keeps the KV cache of text it has already processed skips
that work, which lowers time-to-first-token. vLLM does this in GPU memory;
LMCache extends it to CPU RAM and local disk, so a prefix that was pushed out
of GPU memory is loaded back instead of recomputed.

Time-to-first-token is measured by replaying a fixed trace
(`bench/trace.jsonl`): the 100 eval conversations with the ideal tool calls
already made, so every setup is timed on exactly the same prompts.

## Repository layout

| Folder | Contents |
|---|---|
| `agent/` | Database, market data, tools, trade guardrails, agent loop, prompts, model backends |
| `eval/` | Question builder, the 100 questions, scorer, runner |
| `bench/` | Fixed trace, time-to-first-token and KV-cache benchmarks |
| `serving/` | Launch scripts for the model servers |
| `notebooks/` | Colab notebooks that run each experiment on a T4 GPU |
| `data/` | Recorded market data snapshot |
| `tests/` | Tests that run without a GPU |

## Running it

The experiments run on a free Colab T4. Open a notebook in Colab, pick the T4
GPU runtime, and choose Runtime > Run all. The first cell clones this repo.

| Notebook | What it does |
|---|---|
| [`00_smoke_test`](https://colab.research.google.com/github/sauerpatchkid/Portfolio_Assistant/blob/main/notebooks/00_smoke_test.ipynb) | Checks that each model server starts and handles a tool call |
| [`01_eval_vllm`](https://colab.research.google.com/github/sauerpatchkid/Portfolio_Assistant/blob/main/notebooks/01_eval_vllm.ipynb) | Eval suite on both models, and full-pass time at 1 to 32 concurrent requests |
| [`02_kv_cache`](https://colab.research.google.com/github/sauerpatchkid/Portfolio_Assistant/blob/main/notebooks/02_kv_cache.ipynb) | Time-to-first-token with KV-cache reuse off, on, and extended to RAM and disk |
| [`03_sglang`](https://colab.research.google.com/github/sauerpatchkid/Portfolio_Assistant/blob/main/notebooks/03_sglang.ipynb) | Same eval suite and trace on the alternative backend |
| [`04_hf_baseline`](https://colab.research.google.com/github/sauerpatchkid/Portfolio_Assistant/blob/main/notebooks/04_hf_baseline.ipynb) | Same suite through sequential in-notebook generation |

Against any running OpenAI-compatible server:

```bash
python -m eval.run_eval --name my_run --model Qwen/Qwen3-8B-AWQ --base-url http://localhost:8000/v1 --concurrency 16
```

Tests, no GPU needed:

```bash
python -m pytest
```

## Results

Not measured yet. This section will be filled in from the Colab runs.

## Limitations

- One T4 GPU and one quantized model; timings will not transfer to other hardware.
- Scoring is keyword- and number-based, so it can miss a correct answer phrased
  unusually, or accept a wrong one that happens to contain the right number.
- The benchmark trace uses ideal tool calls, not a model's own.
- The portfolio is mock data and the market data is a snapshot from one day.
