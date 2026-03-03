from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_student.csv"

PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUT_MODEL_TABLE = TABLES_DIR / "linear_model_comparison.csv"
OUT_COEFS_ALL = TABLES_DIR / "linear_coefficients_all.csv"
OUT_COEFS_TOP = TABLES_DIR / "linear_coefficients_top20.csv"

OUT_PRED_VS_TRUE = PLOTS_DIR / "linear_pred_vs_true.png"
OUT_RESIDUALS = PLOTS_DIR / "linear_residuals.png"
OUT_COEF_SHRINK = PLOTS_DIR / "ridge_lasso_shrinkage_top20.png"


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def extract_coefficients(pipe: Pipeline, feature_names: list[str]) -> pd.Series:
    reg = pipe.named_steps["regressor"]
    if not hasattr(reg, "coef_"):
        raise ValueError("Regressor has no coef_ attribute.")
    return pd.Series(reg.coef_, index=feature_names)


def main() -> None:
    ensure_dirs()

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset at: {CLEAN_PATH}. Run scripts/01_preprocess.py first."
        )

    df = pd.read_csv(CLEAN_PATH)
    if "G3" not in df.columns:
        raise ValueError("clean_student.csv must contain 'G3'.")

    y = df["G3"].to_numpy()
    X = df.drop(columns=["G3", "high_performer"], errors="ignore")
    feature_names = list(X.columns)

    # Fixed split for reproducible comparisons
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 1) OLS baseline (with StandardScaler)
    ols_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    ols_cv = cross_val_score(
        ols_pipe, X_train, y_train, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    ols_cv_rmse = float(-ols_cv.mean())

    ols_pipe.fit(X_train, y_train)
    ols_pred = ols_pipe.predict(X_test)
    ols_metrics = eval_regression(y_test, ols_pred)

    # 2) Ridge tuning
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(random_state=42))
    ])
    ridge_grid = {"regressor__alpha": np.logspace(-4, 4, 25)}
    ridge_search = GridSearchCV(
        ridge_pipe, ridge_grid, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    ridge_search.fit(X_train, y_train)
    ridge_best = ridge_search.best_estimator_
    ridge_pred = ridge_best.predict(X_test)
    ridge_metrics = eval_regression(y_test, ridge_pred)

    # 3) Lasso tuning
    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Lasso(max_iter=20000, random_state=42))
    ])
    lasso_grid = {"regressor__alpha": np.logspace(-4, 1, 25)}
    lasso_search = GridSearchCV(
        lasso_pipe, lasso_grid, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    lasso_search.fit(X_train, y_train)
    lasso_best = lasso_search.best_estimator_
    lasso_pred = lasso_best.predict(X_test)
    lasso_metrics = eval_regression(y_test, lasso_pred)

    # Save comparison table
    results = pd.DataFrame([
        {
            "model": "OLS (Standardized)",
            "cv_rmse": ols_cv_rmse,
            "test_rmse": ols_metrics["rmse"],
            "test_mae": ols_metrics["mae"],
            "test_r2": ols_metrics["r2"],
            "best_alpha": np.nan
        },
        {
            "model": "Ridge (tuned)",
            "cv_rmse": float(-ridge_search.best_score_),
            "test_rmse": ridge_metrics["rmse"],
            "test_mae": ridge_metrics["mae"],
            "test_r2": ridge_metrics["r2"],
            "best_alpha": float(ridge_search.best_params_["regressor__alpha"])
        },
        {
            "model": "Lasso (tuned)",
            "cv_rmse": float(-lasso_search.best_score_),
            "test_rmse": lasso_metrics["rmse"],
            "test_mae": lasso_metrics["mae"],
            "test_r2": lasso_metrics["r2"],
            "best_alpha": float(lasso_search.best_params_["regressor__alpha"])
        },
    ])
    results.to_csv(OUT_MODEL_TABLE, index=False)

    # Coefficients for interpretation
    ols_coefs = extract_coefficients(ols_pipe, feature_names).rename("ols")
    ridge_coefs = extract_coefficients(ridge_best, feature_names).rename("ridge")
    lasso_coefs = extract_coefficients(lasso_best, feature_names).rename("lasso")

    coef_df = pd.concat([ols_coefs, ridge_coefs, lasso_coefs], axis=1)
    coef_df["abs_ols"] = coef_df["ols"].abs()
    coef_df = coef_df.sort_values("abs_ols", ascending=False)

    coef_df.to_csv(OUT_COEFS_ALL, index=True)

    top20 = coef_df.head(20).drop(columns=["abs_ols"])
    top20.to_csv(OUT_COEFS_TOP, index=True)

    # Shrinkage plot (Top 20 by |OLS|)
    fig = plt.figure()
    y_pos = np.arange(len(top20.index))
    width = 0.25
    plt.barh(y_pos - width, top20["ols"].values, height=0.25, label="OLS")
    plt.barh(y_pos, top20["ridge"].values, height=0.25, label="Ridge")
    plt.barh(y_pos + width, top20["lasso"].values, height=0.25, label="Lasso")
    plt.yticks(y_pos, top20.index)
    plt.xlabel("Standardized Coefficient")
    plt.title("Coefficient Shrinkage (Top 20 by |OLS|)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(OUT_COEF_SHRINK, dpi=200)
    plt.close(fig)

    # Diagnostics plots (best by test RMSE)
    best_name = results.sort_values("test_rmse").iloc[0]["model"]
    if best_name == "OLS (Standardized)":
        yhat = ols_pred
    elif best_name == "Ridge (tuned)":
        yhat = ridge_pred
    else:
        yhat = lasso_pred

    fig = plt.figure()
    plt.scatter(y_test, yhat, s=12)
    plt.xlabel("True G3")
    plt.ylabel("Predicted G3")
    plt.title(f"Predicted vs True (Best Linear: {best_name})")
    fig.tight_layout()
    fig.savefig(OUT_PRED_VS_TRUE, dpi=200)
    plt.close(fig)

    resid = y_test - yhat
    fig = plt.figure()
    plt.scatter(yhat, resid, s=12)
    plt.axhline(0)
    plt.xlabel("Predicted G3")
    plt.ylabel("Residual (True - Pred)")
    plt.title(f"Residual Plot (Best Linear: {best_name})")
    fig.tight_layout()
    fig.savefig(OUT_RESIDUALS, dpi=200)
    plt.close(fig)

    print("Saved:")
    print(" - Model comparison:", OUT_MODEL_TABLE)
    print(" - Coefficients (all):", OUT_COEFS_ALL)
    print(" - Coefficients (top 20):", OUT_COEFS_TOP)
    print(" - Shrinkage plot:", OUT_COEF_SHRINK)
    print(" - Pred vs True:", OUT_PRED_VS_TRUE)
    print(" - Residuals:", OUT_RESIDUALS)
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()