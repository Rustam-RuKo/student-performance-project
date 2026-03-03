from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "student-mat.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_CLEAN = PROCESSED_DIR / "clean_student.csv"

RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
OUT_DICT = RESULTS_TABLES / "data_dictionary.csv"
OUT_OVERVIEW = RESULTS_TABLES / "data_overview.csv"

CUTOFF = 14  # high_performer threshold for G3


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw dataset at: {RAW_PATH}\n"
            "Place 'student-mat.csv' (semicolon-separated) into data/raw/."
        )
    return pd.read_csv(RAW_PATH, sep=";")


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleaning that stays faithful to the original dataset:
    - drop duplicates
    - coerce grade columns to numeric
    - fill missing numeric with median, categorical with mode
    """
    df = df.drop_duplicates().reset_index(drop=True)

    for col in ["G1", "G2", "G3"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    if "G3" not in df.columns:
        raise ValueError("Expected column 'G3' (final grade) not found.")
    df = df.copy()
    df["high_performer"] = (df["G3"] >= CUTOFF).astype(int)
    return df


def encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a single modeling table:
    - keeps targets: G3 (regression) and high_performer (classification)
    - one-hot encodes categorical features with drop_first=True
    """
    target_cols = ["G3", "high_performer"]
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required target columns: {missing}")

    X = df.drop(columns=target_cols)
    X_encoded = pd.get_dummies(X, drop_first=True)
    out = pd.concat([df[target_cols], X_encoded], axis=1)
    return out


def write_data_dictionary(raw_df: pd.DataFrame) -> None:
    dictionary = {
        "G1": "First period grade (0–20)",
        "G2": "Second period grade (0–20)",
        "G3": "Final grade (0–20)",
        "high_performer": f"Binary label: 1 if G3 >= {CUTOFF}, else 0",
        "age": "Student age",
        "studytime": "Weekly study time (ordinal buckets)",
        "failures": "Past class failures",
        "absences": "Number of school absences",
        "Medu/Fedu": "Mother/Father education (ordinal)",
        "Mjob/Fjob": "Mother/Father job category",
        "schoolsup/famsup": "Extra educational support / family support (yes/no)",
        "higher": "Wants to take higher education (yes/no)",
        "internet": "Internet access at home (yes/no)",
        "romantic": "In a romantic relationship (yes/no)",
        "health": "Current health status (ordinal)",
    }

    rows = []
    for col in raw_df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(raw_df[col].dtype),
                "description": dictionary.get(col, ""),
                "n_unique": int(raw_df[col].nunique(dropna=True)),
                "example_values": ", ".join(map(str, raw_df[col].dropna().unique()[:5])),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DICT, index=False)


def write_overview(df: pd.DataFrame) -> None:
    overview = pd.DataFrame(
        {
            "n_rows": [len(df)],
            "n_features_total_including_targets": [df.shape[1]],
            "n_features_excluding_targets": [
                df.drop(columns=["G3", "high_performer"], errors="ignore").shape[1]
            ],
            "n_missing_total": [int(df.isna().sum().sum())],
            "high_performer_rate": [
                float(df["high_performer"].mean()) if "high_performer" in df.columns else np.nan
            ],
        }
    )
    overview.to_csv(OUT_OVERVIEW, index=False)


def main() -> None:
    ensure_dirs()

    raw = load_raw()
    cleaned = basic_clean(raw)
    with_targets = add_targets(cleaned)
    model_df = encode_for_model(with_targets)

    model_df.to_csv(OUT_CLEAN, index=False)

    write_data_dictionary(raw)
    write_overview(model_df)

    print("Saved:")
    print(" - Clean modeling dataset:", OUT_CLEAN)
    print(" - Data dictionary:", OUT_DICT)
    print(" - Data overview:", OUT_OVERVIEW)


if __name__ == "__main__":
    main()