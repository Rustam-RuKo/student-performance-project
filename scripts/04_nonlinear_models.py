from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_student.csv"

PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

# Existing outputs (keep for compatibility)
OUT_DT_TUNING_PLOT = PLOTS_DIR / "dt_tuning_curve_rmse_vs_depth.png"
OUT_DT_IMPORTANCE_PLOT = PLOTS_DIR / "dt_feature_importance.png"
OUT_RESULTS_TABLE = TABLES_DIR / "baseline_vs_tree_rmse.csv"

# New Random Forest outputs
OUT_RF_IMPORTANCE_PLOT = PLOTS_DIR / "rf_top10_importances.png"
OUT_RF_IMPORTANCE_CSV = TABLES_DIR / "rf_feature_importance_top20.csv"


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

    # Deliverable asks for test RMSE, so we keep a hold-out test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------
    # 1) Baseline model (Linear Regression)
    # -------------------------
    baseline = LinearRegression()
    baseline_cv = GridSearchCV(
        baseline,
        param_grid={},  # no tuning; we use GridSearchCV to compute CV RMSE consistently
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    baseline_cv.fit(X_train, y_train)
    baseline_cv_rmse = -baseline_cv.best_score_

    baseline.fit(X_train, y_train)
    baseline_test_rmse = rmse(y_test, baseline.predict(X_test))

    # -------------------------
    # 2) Nonlinear model A: Decision Tree (tuned)
    # -------------------------
    tree = DecisionTreeRegressor(random_state=42)

    dt_param_grid = {
        "max_depth": list(range(1, 21)),
        "min_samples_leaf": [1, 2, 5, 10, 20],
    }

    dt_search = GridSearchCV(
        tree,
        param_grid=dt_param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    dt_search.fit(X_train, y_train)

    best_tree = dt_search.best_estimator_
    tree_cv_rmse = -dt_search.best_score_
    tree_test_rmse = rmse(y_test, best_tree.predict(X_test))

    # ---- Decision Tree tuning curve plot
    dt_results = pd.DataFrame(dt_search.cv_results_)
    dt_results["rmse"] = -dt_results["mean_test_score"]
    curve = dt_results.groupby("param_max_depth", as_index=False)["rmse"].min()
    curve = curve.sort_values("param_max_depth")

    fig = plt.figure()
    plt.plot(curve["param_max_depth"], curve["rmse"])
    plt.xlabel("max_depth")
    plt.ylabel("CV RMSE (lower is better)")
    plt.title("Decision Tree Tuning Curve (CV RMSE vs max_depth)")
    fig.tight_layout()
    fig.savefig(OUT_DT_TUNING_PLOT, dpi=200)
    plt.close(fig)

    # ---- Decision Tree feature importance plot
    dt_importances = pd.Series(best_tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    dt_top = dt_importances.head(15).iloc[::-1]  # reverse for readable barh

    fig = plt.figure()
    plt.barh(dt_top.index, dt_top.values)
    plt.xlabel("Importance")
    plt.title("Decision Tree Feature Importances (Top 15)")
    fig.tight_layout()
    fig.savefig(OUT_DT_IMPORTANCE_PLOT, dpi=200)
    plt.close(fig)

    # -------------------------
    # 3) Nonlinear model B: Random Forest (tuned)
    # -------------------------
    rf = RandomForestRegressor(random_state=42)

    # Small grid on purpose: fast + strong enough for this dataset
    rf_param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.5],  # random feature subset per split
    }

    rf_search = GridSearchCV(
        rf,
        param_grid=rf_param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_
    rf_cv_rmse = -rf_search.best_score_
    rf_test_rmse = rmse(y_test, best_rf.predict(X_test))

    # ---- Random Forest feature importance CSV + plot
    rf_importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Save Top-20 importances to CSV
    rf_top20 = rf_importances.head(20).reset_index()
    rf_top20.columns = ["feature", "importance"]
    rf_top20.to_csv(OUT_RF_IMPORTANCE_CSV, index=False)

    # Plot Top-10 importances
    rf_top10 = rf_importances.head(10).iloc[::-1]
    fig = plt.figure()
    plt.barh(rf_top10.index, rf_top10.values)
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importances (Top 10)")
    fig.tight_layout()
    fig.savefig(OUT_RF_IMPORTANCE_PLOT, dpi=200)
    plt.close(fig)

    # -------------------------
    # Results table: baseline vs nonlinear (CV RMSE, test RMSE)
    # (We keep the same filename but add Random Forest as a new row.)
    # -------------------------
    out = pd.DataFrame(
        [
            {"model": "Baseline (LinearRegression)", "cv_rmse": baseline_cv_rmse, "test_rmse": baseline_test_rmse},
            {"model": "Nonlinear (Tuned DecisionTree)", "cv_rmse": tree_cv_rmse, "test_rmse": tree_test_rmse},
            {"model": "Nonlinear (Tuned RandomForest)", "cv_rmse": rf_cv_rmse, "test_rmse": rf_test_rmse},
        ]
    )
    out.to_csv(OUT_RESULTS_TABLE, index=False)

    print("Best tree params:", dt_search.best_params_)
    print("Best RF params:", rf_search.best_params_)
    print(out)


if __name__ == "__main__":
    main()