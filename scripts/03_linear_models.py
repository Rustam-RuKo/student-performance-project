from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_student.csv"

PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUT_COEF_PLOT = PLOTS_DIR / "linear_top10_ols_coefficients.png"
OUT_COEF_TABLE = TABLES_DIR / "ols_all_coefficients.csv"
OUT_TOP5_TABLE = TABLES_DIR / "ols_top5_coefficients.csv"
OUT_MODEL_TABLE = TABLES_DIR / "linear_model_comparison.csv"


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    ensure_dirs()

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Missing clean dataset at: {CLEAN_PATH}. Run scripts/01_preprocess.py first."
        )

    df = pd.read_csv(CLEAN_PATH)

    if "G3" not in df.columns:
        raise ValueError("clean_student.csv must contain 'G3'.")

    y = df["G3"].to_numpy()
    X = df.drop(columns=["G3", "high_performer"], errors="ignore")

    # -------------------------
    # Train/Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Pipeline: Standardization + Linear Regression
    # -------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    # -------------------------
    # Cross-validation (CV RMSE)
    # -------------------------
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    cv_rmse = -cv_scores.mean()

    # -------------------------
    # Fit on full training set
    # -------------------------
    model.fit(X_train, y_train)

    # -------------------------
    # Test RMSE
    # -------------------------
    y_pred_test = model.predict(X_test)
    test_rmse = rmse(y_test, y_pred_test)

    # -------------------------
    # Extract standardized coefficients
    # -------------------------
    regressor = model.named_steps["regressor"]
    coefficients = pd.Series(regressor.coef_, index=X.columns)

    coef_df = coefficients.sort_values(ascending=False)
    coef_df.to_csv(OUT_COEF_TABLE)

    top5 = coef_df.abs().sort_values(ascending=False).head(5)
    top5.to_csv(OUT_TOP5_TABLE)

    # -------------------------
    # Plot top 10 standardized coefficients
    # -------------------------
    top10 = coef_df.abs().sort_values(ascending=False).head(10)
    top10_signed = coefficients.loc[top10.index]

    fig = plt.figure()
    plt.barh(top10_signed.index[::-1], top10_signed.values[::-1])
    plt.xlabel("Standardized Coefficient")
    plt.title("Top 10 OLS Coefficients (Standardized Features)")
    fig.tight_layout()
    fig.savefig(OUT_COEF_PLOT, dpi=200)
    plt.close(fig)

    # -------------------------
    # Save model performance
    # -------------------------
    performance_df = pd.DataFrame([
        {
            "model": "Linear Regression (Standardized)",
            "cv_rmse": cv_rmse,
            "test_rmse": test_rmse
        }
    ])

    performance_df.to_csv(OUT_MODEL_TABLE, index=False)

    print("Linear Regression (with normalization)")
    print("CV RMSE:", cv_rmse)
    print("Test RMSE:", test_rmse)


if __name__ == "__main__":
    main()