# src/analyze_results.py
"""Utility script to inspect the JSON outputs of `train_flow_gpu.py`.

Usage
-----
python src/analyze_results.py --dir results_cpu

It will print a sorted table of metrics and, if matplotlib is available,
render a horizontal bar chart of F1 scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def load_metrics(dir_path: Path) -> pd.DataFrame:
    metrics_path = dir_path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with open(metrics_path, "r", encoding="utf-8") as fp:
        data: Dict[str, Dict[str, float]] = json.load(fp)
    df = pd.DataFrame.from_dict(data, orient="index")
    return df


def main():
    ap = argparse.ArgumentParser(description="Analyze results directory")
    ap.add_argument("--dir", default="results_cpu", help="results directory path")
    args = ap.parse_args()

    dir_path = Path(args.dir)
    df = load_metrics(dir_path)

    print("\n=== Metrics (sorted by F1) ===")
    print(df.sort_values("f1", ascending=False).to_markdown(floatfmt=".4f"))

    sfe_feat = dir_path / "sfe_selected_features.json"
    if sfe_feat.exists():
        with sfe_feat.open("r", encoding="utf-8") as fp:
            info = json.load(fp)
        print("\nS_FE selected features ({} features, val_F1 {:.4f}):".format(
            len(info["selected"]), info["best_val"]
        ))
        print(", ".join(info["selected"]))

    if HAS_PLOT:
        df.sort_values("f1").plot.barh(y="f1", legend=False, figsize=(6, 3))
        plt.xlim(0, 1)
        plt.title("Model F1 scores")
        plt.tight_layout()
        plt.show()
    else:
        print("(matplotlib not installed → skipping plot)")


if __name__ == "__main__":
    main()
