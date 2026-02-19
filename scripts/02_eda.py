from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "student-mat.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_student.csv"

PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUT_SUMMARY = TABLES_DIR / "eda_summary.csv"
OUT_GROUPS = TABLES_DIR / "eda_group_means.csv"


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_for_readable_plots() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw dataset at: {RAW_PATH}")
    return pd.read_csv(RAW_PATH, sep=";")


def save_plot(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dirs()

    raw = load_raw_for_readable_plots()

    needed = ["G3", "studytime", "absences", "goout"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw dataset missing columns needed for EDA: {missing}")

    # Summary stats table
    summary = raw[["G3", "studytime", "absences", "goout"]].describe().T
    summary.to_csv(OUT_SUMMARY)

    # Group means for quick interpretation
    group_means = (
        raw.groupby("studytime", as_index=False)["G3"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(columns={"mean": "G3_mean", "median": "G3_median"})
    )
    group_means.to_csv(OUT_GROUPS, index=False)

    # Plot 1: Histogram of G3
    fig = plt.figure()
    plt.hist(raw["G3"], bins=10)
    plt.title("Distribution of Final Grade (G3)")
    plt.xlabel("G3")
    plt.ylabel("Count")
    save_plot(fig, PLOTS_DIR / "eda_G3_hist.png")

    # Plot 2: Histogram of absences
    fig = plt.figure()
    plt.hist(raw["absences"], bins=15)
    plt.title("Distribution of Absences")
    plt.xlabel("Absences")
    plt.ylabel("Count")
    save_plot(fig, PLOTS_DIR / "eda_absences_hist.png")

    # Plot 3: Boxplot of G3 by studytime
    fig = plt.figure()
    cats = sorted(raw["studytime"].unique())
    data = [raw.loc[raw["studytime"] == k, "G3"] for k in cats]
    plt.boxplot(data, labels=cats)
    plt.title("G3 by Weekly Study Time Category")
    plt.xlabel("studytime (1:<2h, 2:2–5h, 3:5–10h, 4:>10h)")
    plt.ylabel("G3")
    save_plot(fig, PLOTS_DIR / "eda_G3_by_studytime_box.png")

    # Plot 4: Boxplot of G3 by goout
    fig = plt.figure()
    go = sorted(raw["goout"].unique())
    data = [raw.loc[raw["goout"] == k, "G3"] for k in go]
    plt.boxplot(data, labels=go)
    plt.title("G3 by Going-Out Frequency (goout)")
    plt.xlabel("goout (1–5)")
    plt.ylabel("G3")
    save_plot(fig, PLOTS_DIR / "eda_G3_by_goout_box.png")

    # Plot 5: Scatter absences vs G3
    fig = plt.figure()
    plt.scatter(raw["absences"], raw["G3"])
    plt.title("Absences vs Final Grade (G3)")
    plt.xlabel("Absences")
    plt.ylabel("G3")
    save_plot(fig, PLOTS_DIR / "eda_absences_vs_G3_scatter.png")

    print(f"Saved summary table: {OUT_SUMMARY}")
    print(f"Saved group means: {OUT_GROUPS}")
    print(f"Saved plots into: {PLOTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
