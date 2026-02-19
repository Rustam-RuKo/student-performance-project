from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_student.csv"

PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUT_COMP = TABLES_DIR / "linear_model_comparison.csv"
OUT_TOP5 = TABLES_DIR / "ols_top5_coefficients.csv"
OUT_COEFS = TABLES_DIR / "ols_all_coefficients.csv"
OUT_PLOT = PLOTS_DIR / "linear_top10_ols_coefficients.png"


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing clean dataset at: {CLEAN_PATH}. Run scripts/01_preprocess.py first.")

    df = pd.read_csv(CLEAN_PATH)

    if "G3" not in df.columns or "high_performer" not in df.columns:
        raise ValueError("clean_student.csv must contain 'G3' and 'high_performer' columns.")

    y = df["G3"].to_numpy()
    X = df.drop(columns=["G3", "high_performer"])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline OLS with cross-validation
    ols = LinearRegression()
    ols_scores = cross_validate(
        ols, X, y, cv=cv,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"}
    )
    ols_mse = -ols_scores["test_mse"].mean()
    ols_r2 = ols_scores["test_r2"].mean()

    # Fit OLS once on full data for coefficient interpretation (interpretation-only)
    ols.fit(X, y)
    ols_coef = pd.Series(ols.coef_, index=X.columns)
    ols_coef_df = ols_coef.sort_values(key=np.abs, ascending=False).rename("coef").to_frame()
    ols_coef_df.to_csv(OUT_COEFS)

    top5 = ols_coef_df.head(5)
    top5.to_csv(OUT_TOP5)

    # Plot top 10 coefficient magnitudes (no custom colors)
    top10 = ols_coef_df.head(10).iloc[::-1]
    fig = plt.figure()
    plt.barh(top10.index, top10["coef"].values)
    plt.title("Top 10 OLS Coefficients (by |magnitude|)")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    fig.savefig(OUT_PLOT, dpi=200)
    plt.close(fig)

    # Ridge with CV model selection (alpha)
    ridge = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", Ridge())
    ])
    ridge_grid = {"model__alpha": np.logspace(-3, 3, 25)}
    ridge_search = GridSearchCV(ridge, ridge_grid, cv=cv, scoring="neg_mean_squared_error")
    ridge_search.fit(X, y)
    ridge_best = ridge_search.best_estimator_
    ridge_alpha = ridge_search.best_params_["model__alpha"]

    ridge_scores = cross_validate(
        ridge_best, X, y, cv=cv,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"}
    )
    ridge_mse = -ridge_scores["test_mse"].mean()
    ridge_r2 = ridge_scores["test_r2"].mean()

    # Lasso with CV model selection (alpha)
    lasso = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", Lasso(max_iter=50000))
    ])
    lasso_grid = {"model__alpha": np.logspace(-3, 1, 25)}
    lasso_search = GridSearchCV(lasso, lasso_grid, cv=cv, scoring="neg_mean_squared_error")
    lasso_search.fit(X, y)
    lasso_best = lasso_search.best_estimator_
    lasso_alpha = lasso_search.best_params_["model__alpha"]

    lasso_scores = cross_validate(
        lasso_best, X, y, cv=cv,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"}
    )
    lasso_mse = -lasso_scores["test_mse"].mean()
    lasso_r2 = lasso_scores["test_r2"].mean()

    comp = pd.DataFrame(
        [
            {"model": "OLS", "cv_mse": ols_mse, "cv_r2": ols_r2, "selected_param": ""},
            {"model": "Ridge", "cv_mse": ridge_mse, "cv_r2": ridge_r2, "selected_param": f"alpha={ridge_alpha:.4g}"},
            {"model": "Lasso", "cv_mse": lasso_mse, "cv_r2": lasso_r2, "selected_param": f"alpha={lasso_alpha:.4g}"},
        ]
    )
    comp.to_csv(OUT_COMP, index=False)

    print(f"Saved model comparison: {OUT_COMP}")
    print(f"Saved OLS coefficients: {OUT_COEFS}")
    print(f"Saved top-5 coefficients: {OUT_TOP5}")
    print(f"Saved coefficient plot: {OUT_PLOT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
