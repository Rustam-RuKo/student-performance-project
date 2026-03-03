from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_student.csv"

PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUT_CORR_TABLE = TABLES_DIR / "top_correlations_with_G3.csv"
OUT_G3_HIST = PLOTS_DIR / "g3_distribution.png"
OUT_G1_G3 = PLOTS_DIR / "g1_vs_g3.png"
OUT_G2_G3 = PLOTS_DIR / "g2_vs_g3.png"
OUT_TOP_CORR_BAR = PLOTS_DIR / "top_correlations_with_G3.png"


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset at: {CLEAN_PATH}. Run scripts/01_preprocess.py first."
        )

    df = pd.read_csv(CLEAN_PATH)

    if "G3" not in df.columns:
        raise ValueError("Expected 'G3' in clean_student.csv")

    # 1) Target distribution
    fig = plt.figure()
    plt.hist(df["G3"], bins=21)
    plt.xlabel("G3 (Final Grade)")
    plt.ylabel("Count")
    plt.title("Distribution of Final Grade (G3)")
    fig.tight_layout()
    fig.savefig(OUT_G3_HIST, dpi=200)
    plt.close(fig)

    # 2) G1/G2 relationships (important since you keep them)
    if "G1" in df.columns:
        fig = plt.figure()
        plt.scatter(df["G1"], df["G3"], s=10)
        plt.xlabel("G1 (First Period Grade)")
        plt.ylabel("G3 (Final Grade)")
        plt.title("G1 vs G3")
        fig.tight_layout()
        fig.savefig(OUT_G1_G3, dpi=200)
        plt.close(fig)

    if "G2" in df.columns:
        fig = plt.figure()
        plt.scatter(df["G2"], df["G3"], s=10)
        plt.xlabel("G2 (Second Period Grade)")
        plt.ylabel("G3 (Final Grade)")
        plt.title("G2 vs G3")
        fig.tight_layout()
        fig.savefig(OUT_G2_G3, dpi=200)
        plt.close(fig)

    # 3) Correlations with G3 (numeric only)
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr(numeric_only=True)["G3"].sort_values(ascending=False)
    corr_table = corr.drop(index=["G3"]).to_frame(name="corr_with_G3")
    corr_table.to_csv(OUT_CORR_TABLE)

    top = corr_table.head(15)
    fig = plt.figure()
    plt.barh(top.index[::-1], top["corr_with_G3"].values[::-1])
    plt.xlabel("Correlation with G3")
    plt.title("Top Numeric Feature Correlations with G3")
    fig.tight_layout()
    fig.savefig(OUT_TOP_CORR_BAR, dpi=200)
    plt.close(fig)

    print("Saved:")
    print(" -", OUT_G3_HIST)
    print(" -", OUT_G1_G3 if "G1" in df.columns else "(G1 missing)")
    print(" -", OUT_G2_G3 if "G2" in df.columns else "(G2 missing)")
    print(" -", OUT_CORR_TABLE)
    print(" -", OUT_TOP_CORR_BAR)


if __name__ == "__main__":
    main()