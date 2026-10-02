# --- Result Figures ---
# Draws the charts in the README from the collected results:
#
#   python -m scripts.make_figures        -> results/figures/*.png
#
# Every number plotted is read from results/*.csv or results/<run>/records.jsonl.
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"      # series 1
ORANGE = "#eb6834"    # series 2

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": INK_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


CATEGORY_LABELS = {
    "lookup": "Single lookups",
    "educational": "Concept explanations (no tools)",
    "afford": "Whole shares affordable with cash",
    "cash_after_buy": "Cash left after a hypothetical buy",
    "sell_to_buy": "Sell one stock to buy another",
    "history_compare": "Compare two stocks' performance",
    "pe_and_shares": "P/E ratios and shares owned",
    "last_purchase": "Last purchase price vs now",
    "price_and_news": "Price and news",
    "closest_to_high": "Closest to its 52-week high",
    "sector_and_cash": "Sector share and cash",
    "shares_and_strike": "Shares owned and option strike",
    "cash_share": "Cash share and what it buys",
    "change_and_value": "Price change and position value",
    "trade": "Trades (request, then confirm)",
}


def read_csv(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def read_records(run):
    with open(RESULTS / run / "records.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def new_axes(width, height, title, subtitle):
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.suptitle(title, x=0.02, y=0.98, ha="left", va="top", fontsize=13, fontweight="semibold")
    fig.text(0.02, 0.98 - 0.30 / height, subtitle, ha="left", va="top", fontsize=9.5, color=INK_SECONDARY)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return fig, ax


def save(fig, name, top):
    FIGURES.mkdir(exist_ok=True)
    fig.subplots_adjust(top=top)
    fig.savefig(FIGURES / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", FIGURES / name)


def accuracy_by_category():
    """Before -> after per question type: one row each, v1 dot and v2 dot."""
    v1 = read_records("agent_v1_8b")
    v2 = {r["id"]: r for r in read_records("agent_v2_8b")}

    categories = []
    for record in v1:
        if record["category"] not in categories:
            categories.append(record["category"])
    rows = []
    for category in categories:
        ids = [r["id"] for r in v1 if r["category"] == category]
        before = sum(r["correct"] for r in v1 if r["category"] == category)
        after = sum(v2[i]["correct"] for i in ids)
        rows.append((CATEGORY_LABELS[category], len(ids), before, after))
    rows.sort(key=lambda row: (row[3] - row[2]) / row[1])

    fig, ax = new_axes(8.6, 6.2, "Correct answers by question type",
                       "Qwen3-8B, original agent (v1) and revised agent (v2). 100 questions.")
    for y, (label, n, before, after) in enumerate(rows):
        ax.plot([100 * before / n, 100 * after / n], [y, y], color=BASELINE, linewidth=2, zorder=2)
        ax.scatter(100 * before / n, y, s=70, color=MUTED, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   clip_on=False)
        ax.scatter(100 * after / n, y, s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4,
                   clip_on=False)
        ax.text(106, y, f"{before} → {after} of {n}", va="center", ha="left", fontsize=9, color=INK_SECONDARY)

    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.set_xlim(-4, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of questions answered correctly")
    ax.legend(
        handles=[Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=8, label="v1"),
                 Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=8, label="v2")],
        loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.0), handletextpad=0.2,
    )
    save(fig, "accuracy_by_category.png", top=0.86)


def horizontal_bars(ax, labels, series, unit, decimals, bar_height=0.32):
    """Thin horizontal bars; with two series they sit one above the other in each row."""
    count = len(series)
    longest = max(max(values) for _, values, _ in series)
    for index, (_, values, color) in enumerate(series):
        offset = (index - (count - 1) / 2) * (bar_height + 0.06)
        positions = [y + offset for y in range(len(labels))]
        ax.barh(positions, values, height=bar_height, color=color, zorder=3)
        for y, value in zip(positions, values):
            ax.text(value + longest * 0.012, y, f"{value:.{decimals}f} {unit}", va="center", ha="left",
                    fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, longest * 1.16)
    if count > 1:
        ax.legend(handles=[Patch(color=color, label=name) for name, _, color in series],
                  loc="lower right", frameon=False, ncol=count, bbox_to_anchor=(1.0, 1.0))


def full_pass_time():
    summary = read_csv("summary.csv")
    runs = [
        ("In-notebook generation,\none request at a time", "eval_hf_8b_v1"),
        ("Model server, 1 at a time", "eval_vllm_8b_v1_c1"),
        ("Model server, 4 concurrent", "eval_vllm_8b_v1_c4"),
        ("Model server, 8 concurrent", "eval_vllm_8b_v1_c8"),
        ("Model server, 16 concurrent", "eval_vllm_8b_v1_c16"),
        ("Model server, 32 concurrent", "eval_vllm_8b_v1_c32"),
    ]
    minutes = [float(summary[run]["wall_s"]) / 60 for _, run in runs]

    fig, ax = new_axes(8.2, 3.9, "Time for one full pass of the eval suite",
                       "100 questions, Qwen3-8B (4-bit), one T4 GPU. Model server is vLLM.")
    horizontal_bars(ax, [label for label, _ in runs], [("", minutes, BLUE)], "min", 1, bar_height=0.5)
    ax.set_xlabel("Minutes")
    save(fig, "full_pass_time.png", top=0.80)


def time_to_first_token():
    ttft = read_csv("ttft_summary.csv")
    setups = [
        ("KV-cache reuse off", "ttft_no_cache"),
        ("vLLM prefix caching\n(GPU memory)", "ttft_prefix_cache"),
        ("vLLM + LMCache", "ttft_lmcache"),
        ("LMCache only\n(CPU RAM)", "ttft_lmcache_only"),
    ]
    first = [float(ttft[run]["first_step_median_s"]) for _, run in setups]
    later = [float(ttft[run]["later_steps_median_s"]) for _, run in setups]

    fig, ax = new_axes(8.2, 4.4, "Time to first token on multi-step tool-calling turns",
                       "Median over 310 requests replayed from a fixed trace. Qwen3-8B (4-bit), one T4 GPU.")
    horizontal_bars(ax, [label for label, _ in setups],
                    [("First step of a turn", first, BLUE), ("Later steps (after a tool call)", later, ORANGE)],
                    "s", 2)
    ax.set_xlabel("Seconds")
    save(fig, "time_to_first_token.png", top=0.78)


def eviction():
    summary = read_csv("eviction_summary.csv")
    phases = [
        ("First visit", "cold_median_s"),
        ("Repeat, still in GPU memory", "warm_median_s"),
        ("Revisit after eviction\nfrom GPU memory", "revisit_median_s"),
        ("Revisit after eviction\nfrom CPU RAM as well", "late_revisit_s"),
    ]
    alone = [float(summary["eviction_prefix_cache"][key]) for _, key in phases]
    with_cache = [float(summary["eviction_lmcache"][key]) for _, key in phases]

    fig, ax = new_axes(8.2, 4.4, "Returning to a conversation after its KV cache was evicted",
                       "Time to first token for a 4,400-token conversation. Qwen3-8B (4-bit), one T4 GPU.")
    horizontal_bars(ax, [label for label, _ in phases],
                    [("vLLM alone", alone, BLUE), ("vLLM + LMCache (RAM and disk tiers)", with_cache, ORANGE)],
                    "s", 2)
    ax.set_xlabel("Seconds")
    save(fig, "eviction.png", top=0.78)


if __name__ == "__main__":
    accuracy_by_category()
    full_pass_time()
    time_to_first_token()
    eviction()
