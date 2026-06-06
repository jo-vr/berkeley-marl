from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def moving_average(values: list[float], window: int) -> list[float]:
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        out.append(sum(values[start : idx + 1]) / (idx - start + 1))
    return out


def load_logs(results_dir: Path) -> dict[str, list[dict]]:
    logs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                logs[row["condition"]].append(row)
    return logs


def plot_metric(logs: dict[str, list[dict]], metric: str, output_path: Path, window: int) -> None:
    plt.figure(figsize=(8, 5))
    for condition, rows in sorted(logs.items()):
        rows = sorted(rows, key=lambda r: (int(r["seed"]), int(r["episode"])))
        episodes = [int(row["episode"]) for row in rows]
        values = [float(row[metric]) for row in rows]
        plt.plot(episodes, moving_average(values, window), label=condition)
    plt.xlabel("Episode")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} Over Training")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    logs = load_logs(results_dir)
    if not logs:
        raise SystemExit(f"No CSV logs found in {results_dir}")
    plot_metric(logs, "return", figures_dir / "returns.png", args.window)
    plot_metric(logs, "success", figures_dir / "success.png", args.window)
    print(f"Wrote figures to {figures_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Berkeley MARL training logs.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--window", type=int, default=25)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
