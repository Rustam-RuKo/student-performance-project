from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_student.csv"

PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUT_LOGIT = TABLES_DIR / "logistic_results.csv"
OUT_CM = TABLES_DIR / "logistic_confusion_matrix.csv"
OUT_RF = TABLES_DIR / "rf_feature_importance_top20.csv"

OUT_ROC_PLOT = PLOTS_DIR / "logistic_roc_curve.png"
OUT_TREE_PLOT = PLOTS_DIR / "decision_tree_depth3.png"
OUT_RF_PLOT = PLOTS_DIR / "rf_top10_importances.png"


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, path: Path) -> None:
    # Manual ROC curve (no seaborn, no custom colors)
    thresholds = np.unique(y_score)[::-1]
    tpr = []
    fpr = []
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    if P == 0 or N == 0:
        raise ValueError("ROC requires both classes to be present.")

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tpr.append(tp / P)
        fpr.append(fp / N)

    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])  # baseline
    plt.title("ROC Curve (Logistic Regression, CV predictions)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dirs()

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing clean dataset at: {CLEAN_PATH}. Run scripts/01_preprocess.py first.")

    df = pd.read_csv(CLEAN_PATH)

    if "high_performer" not in df.columns or "G3" not in df.columns:
        raise ValueError("clean_student.csv must contain 'G3' and 'high_performer'.")

    y = df["high_performer"].to_numpy()
    X = df.drop(columns=["G3", "high_performer"])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic regression with CV model selection (C) and AUC scoring
    logit = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(max_iter=50000))
    ])
    grid = {"model__C": np.logspace(-3, 3, 25)}
    search = GridSearchCV(logit, grid, cv=cv, scoring="roc_auc")
    search.fit(X, y)
    best_model = search.best_estimator_
    best_C = search.best_params_["model__C"]

    # Cross-validated predicted probabilities for evaluation (not training-set probabilities)
    proba = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)

    # Confusion matrix at threshold 0.5 (simple baseline)
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)

    pd.DataFrame(
        [{
            "best_C": best_C,
            "cv_auc": auc,
            "threshold": 0.5,
            "accuracy": acc,
            "precision": prec,
            "recall": rec
        }]
    ).to_csv(OUT_LOGIT, index=False)

    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(OUT_CM)

    plot_roc_curve(y, proba, OUT_ROC_PLOT)

    # Interpretable decision tree (depth=3)
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    fig = plt.figure(figsize=(14, 7))
    plot_tree(tree, feature_names=X.columns, class_names=["lower", "high"], filled=False)
    plt.title("Decision Tree (max_depth=3)")
    fig.tight_layout()
    fig.savefig(OUT_TREE_PLOT, dpi=200)
    plt.close(fig)

    # Random forest for feature importance comparison (not the main “interpretable” model)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.head(20).rename("importance").to_frame().to_csv(OUT_RF)

    top10 = importances.head(10).iloc[::-1]
    fig = plt.figure()
    plt.barh(top10.index, top10.values)
    plt.title("Top 10 Random Forest Feature Importances")
    plt.xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUT_RF_PLOT, dpi=200)
    plt.close(fig)

    print(f"Saved logistic results: {OUT_LOGIT}")
    print(f"Saved confusion matrix: {OUT_CM}")
    print(f"Saved ROC plot: {OUT_ROC_PLOT}")
    print(f"Saved decision tree plot: {OUT_TREE_PLOT}")
    print(f"Saved RF importances: {OUT_RF}")
    print(f"Saved RF plot: {OUT_RF_PLOT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
