from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_student.csv"

PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUT_TUNING_PLOT = PLOTS_DIR / "dt_tuning_curve_rmse_vs_depth.png"
OUT_IMPORTANCE_PLOT = PLOTS_DIR / "dt_feature_importance.png"
OUT_RESULTS_TABLE = TABLES_DIR / "baseline_vs_tree_rmse.csv"


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    ensure_dirs()

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing clean dataset at: {CLEAN_PATH}. Run scripts/01_preprocess.py first.")

    df = pd.read_csv(CLEAN_PATH)

    if "G3" not in df.columns:
        raise ValueError("clean_student.csv must contain 'G3'.")

    y = df["G3"].to_numpy()
    X = df.drop(columns=["G3", "high_performer"], errors="ignore")

    # You need a test set because the deliverable asks for test RMSE.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------
    # Baseline model (Linear Regression)
    # -------------------------
    baseline = LinearRegression()
    baseline_cv = GridSearchCV(
        baseline,
        param_grid={},  # no tuning, but lets us compute CV RMSE consistently
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )
    baseline_cv.fit(X_train, y_train)
    baseline_cv_rmse = -baseline_cv.best_score_

    baseline.fit(X_train, y_train)
    baseline_test_rmse = rmse(y_test, baseline.predict(X_test))

    # -------------------------
    # Nonlinear model (DecisionTreeRegressor) + tuning
    # -------------------------
    tree = DecisionTreeRegressor(random_state=42)

    param_grid = {
        "max_depth": list(range(1, 21)),
        "min_samples_leaf": [1, 2, 5, 10, 20]
    }

    search = GridSearchCV(
        tree,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_tree = search.best_estimator_
    tree_cv_rmse = -search.best_score_
    tree_test_rmse = rmse(y_test, best_tree.predict(X_test))

    # -------------------------
    # Figure 1: Tuning curve (CV RMSE vs max_depth)
    # We plot the best (lowest) CV RMSE achieved at each depth across min_samples_leaf.
    # -------------------------
    results = pd.DataFrame(search.cv_results_)
    results["rmse"] = -results["mean_test_score"]
    # For each max_depth, keep the best RMSE (min over min_samples_leaf)
    curve = results.groupby("param_max_depth", as_index=False)["rmse"].min()
    curve = curve.sort_values("param_max_depth")

    fig = plt.figure()
    plt.plot(curve["param_max_depth"], curve["rmse"])
    plt.xlabel("max_depth")
    plt.ylabel("CV RMSE (lower is better)")
    plt.title("Decision Tree Tuning Curve (CV RMSE vs max_depth)")
    fig.tight_layout()
    fig.savefig(OUT_TUNING_PLOT, dpi=200)
    plt.close(fig)

    # -------------------------
    # Figure 2: Feature importance bar chart
    # -------------------------
    importances = pd.Series(best_tree.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    top = importances.head(15).iloc[::-1]  # horizontal bar, readable
    fig = plt.figure()
    plt.barh(top.index, top.values)
    plt.xlabel("Importance")
    plt.title("Decision Tree Feature Importances (Top 15)")
    fig.tight_layout()
    fig.savefig(OUT_IMPORTANCE_PLOT, dpi=200)
    plt.close(fig)

    # -------------------------
    # Results table: baseline vs nonlinear (CV RMSE, test RMSE)
    # -------------------------
    out = pd.DataFrame([
        {"model": "Baseline (LinearRegression)", "cv_rmse": baseline_cv_rmse, "test_rmse": baseline_test_rmse},
        {"model": "Nonlinear (Tuned DecisionTree)", "cv_rmse": tree_cv_rmse, "test_rmse": tree_test_rmse},
    ])
    out.to_csv(OUT_RESULTS_TABLE, index=False)

    print("Best tree params:", search.best_params_)
    print(out)


if __name__ == "__main__":
    main()