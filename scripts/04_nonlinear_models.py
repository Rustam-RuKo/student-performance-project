from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_student.csv"

PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUT_MODEL_TABLE = TABLES_DIR / "nonlinear_model_comparison.csv"
OUT_RF_IMPORTANCE = TABLES_DIR / "rf_feature_importance_top20.csv"
OUT_TREE_IMPORTANCE = TABLES_DIR / "tree_feature_importance_top20.csv"

OUT_RF_IMPORTANCE_PLOT = PLOTS_DIR / "rf_feature_importance_top20.png"
OUT_TREE_IMPORTANCE_PLOT = PLOTS_DIR / "tree_feature_importance_top20.png"
OUT_NONLINEAR_PRED_VS_TRUE = PLOTS_DIR / "nonlinear_pred_vs_true.png"


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


def top_importance(model, feature_names: list[str], k: int = 20) -> pd.Series:
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model has no feature_importances_.")
    imp = pd.Series(model.feature_importances_, index=feature_names)
    return imp.sort_values(ascending=False).head(k)


def plot_importance(imp: pd.Series, out_path: Path, title: str) -> None:
    fig = plt.figure()
    plt.barh(imp.index[::-1], imp.values[::-1])
    plt.xlabel("Feature Importance")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


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

    # Same split as linear script
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline: standardized linear regression (fair baseline)
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    baseline_cv = cross_val_score(
        baseline, X_train, y_train, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    baseline_cv_rmse = float(-baseline_cv.mean())
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    base_metrics = eval_regression(y_test, base_pred)

    # Decision Tree tuning
    tree = DecisionTreeRegressor(random_state=42)
    tree_grid = {
        "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
    }
    tree_search = GridSearchCV(
        tree, tree_grid, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    tree_search.fit(X_train, y_train)
    tree_best = tree_search.best_estimator_
    tree_pred = tree_best.predict(X_test)
    tree_metrics = eval_regression(y_test, tree_pred)

    # Random Forest tuning
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.5, 1.0],
    }
    rf_search = GridSearchCV(
        rf, rf_grid, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_pred = rf_best.predict(X_test)
    rf_metrics = eval_regression(y_test, rf_pred)

    # Save comparison table
    results = pd.DataFrame([
        {
            "model": "Baseline OLS (Standardized)",
            "cv_rmse": baseline_cv_rmse,
            "test_rmse": base_metrics["rmse"],
            "test_mae": base_metrics["mae"],
            "test_r2": base_metrics["r2"],
            "best_params": ""
        },
        {
            "model": "Decision Tree (tuned)",
            "cv_rmse": float(-tree_search.best_score_),
            "test_rmse": tree_metrics["rmse"],
            "test_mae": tree_metrics["mae"],
            "test_r2": tree_metrics["r2"],
            "best_params": str(tree_search.best_params_)
        },
        {
            "model": "Random Forest (tuned)",
            "cv_rmse": float(-rf_search.best_score_),
            "test_rmse": rf_metrics["rmse"],
            "test_mae": rf_metrics["mae"],
            "test_r2": rf_metrics["r2"],
            "best_params": str(rf_search.best_params_)
        },
    ])
    results.to_csv(OUT_MODEL_TABLE, index=False)

    # Feature importance (Tree + RF)
    tree_imp = top_importance(tree_best, feature_names, k=20)
    rf_imp = top_importance(rf_best, feature_names, k=20)

    tree_imp.to_frame("importance").to_csv(OUT_TREE_IMPORTANCE)
    rf_imp.to_frame("importance").to_csv(OUT_RF_IMPORTANCE)

    plot_importance(tree_imp, OUT_TREE_IMPORTANCE_PLOT, "Decision Tree Feature Importance (Top 20)")
    plot_importance(rf_imp, OUT_RF_IMPORTANCE_PLOT, "Random Forest Feature Importance (Top 20)")

    # Pred vs True plot for best model (by test RMSE)
    best_name = results.sort_values("test_rmse").iloc[0]["model"]
    if best_name == "Baseline OLS (Standardized)":
        yhat = base_pred
    elif best_name == "Decision Tree (tuned)":
        yhat = tree_pred
    else:
        yhat = rf_pred

    fig = plt.figure()
    plt.scatter(y_test, yhat, s=12)
    plt.xlabel("True G3")
    plt.ylabel("Predicted G3")
    plt.title(f"Predicted vs True (Best Nonlinear: {best_name})")
    fig.tight_layout()
    fig.savefig(OUT_NONLINEAR_PRED_VS_TRUE, dpi=200)
    plt.close(fig)

    print("Saved:")
    print(" - Model comparison:", OUT_MODEL_TABLE)
    print(" - Tree importance table:", OUT_TREE_IMPORTANCE)
    print(" - RF importance table:", OUT_RF_IMPORTANCE)
    print(" - Tree importance plot:", OUT_TREE_IMPORTANCE_PLOT)
    print(" - RF importance plot:", OUT_RF_IMPORTANCE_PLOT)
    print(" - Pred vs True:", OUT_NONLINEAR_PRED_VS_TRUE)
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()