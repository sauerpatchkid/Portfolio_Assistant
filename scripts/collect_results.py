# --- Collect Colab Results ---
# Each Colab notebook downloads a zip of its results/ folder. This merges those
# zips (dropped into results/) into one results/ tree:
#
#   python -m scripts.collect_results
#
# Zips are read oldest first, so a re-run of a notebook replaces its earlier
# rows and files. Smoke-test runs, server logs and databases are left out.
import csv
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SUMMARIES = ["summary.csv", "ttft_summary.csv", "eviction_summary.csv"]
SKIP_PARTS = {"logs", "smoke"}


def wanted(relative: Path) -> bool:
    if relative.name in SUMMARIES or relative.suffix in {".db", ".zip"} or relative.name == ".gitkeep":
        return False
    if SKIP_PARTS & set(relative.parts):
        return False
    return not relative.parts[0].startswith("smoke")


def main():
    zips = sorted(RESULTS.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    rows = {name: {} for name in SUMMARIES}   # summary file -> {run name: row}
    copied = 0

    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if member.endswith("/") or not member.startswith("results/"):
                    continue
                relative = Path(member).relative_to("results")

                if relative.name in SUMMARIES and len(relative.parts) == 1:
                    text = archive.read(member).decode("utf-8")
                    for row in csv.DictReader(io.StringIO(text)):
                        if not row["name"].startswith("smoke"):
                            rows[relative.name][row["name"]] = row
                elif wanted(relative):
                    target = RESULTS / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))
                    copied += 1

    for name, by_run in rows.items():
        if not by_run:
            continue
        columns = []
        for row in by_run.values():
            columns += [c for c in row if c not in columns]
        with open(RESULTS / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(by_run.values())
        print(f"{name}: {len(by_run)} runs")

    print(f"Copied {copied} result files from {len(zips)} zips")


if __name__ == "__main__":
    main()
