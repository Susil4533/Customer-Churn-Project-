"""
modeling.py
-----------
Shared utilities for training, evaluating, and saving churn prediction models.
Used by all three modeling notebooks (03a, 03b, 03c).

Class imbalance strategy: class_weight='balanced'
    - Churn dataset is ~26.6% positive (churners)
    - 'balanced' automatically adjusts weights inversely proportional to class frequency
    - This penalizes the model more for missing churners (the costly mistake)
    - Chosen over SMOTE for simplicity and reproducibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)


# ── 1. Data splitting ────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split featured dataset into train/test sets.
    Stratified split to preserve churn ratio in both sets.
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ── 2. Model evaluation ──────────────────────────────────────────────────────

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a trained model and return key metrics.

    Priority metrics (from blueprint):
    1. ROC-AUC  — overall ranking ability
    2. Recall   — catching actual churners (business priority)
    3. Precision, F1, Accuracy — supporting context
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
    }
    return metrics


# ── 3. Confusion matrix plot ─────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Model"):
    """Plot annotated confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted: Stay", "Predicted: Churn"],
        yticklabels=["Actual: Stay", "Actual: Churn"],
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"  True Positives  (caught churners):  {tp}")
    print(f"  False Negatives (missed churners):  {fn}  ← business cost")
    print(f"  False Positives (wrong alerts):     {fp}")
    print(f"  True Negatives  (correct stays):    {tn}")


# ── 4. ROC curve plot ────────────────────────────────────────────────────────

def plot_roc_curve(model, X_test, y_test, model_name: str = "Model"):
    """Plot ROC curve with AUC score."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ── 5. Metrics summary table ─────────────────────────────────────────────────

def print_metrics_table(metrics: dict, model_name: str = "Model"):
    """Print metrics in a clean readable format."""
    print(f"\n{'='*40}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*40}")
    for metric, value in metrics.items():
        marker = "  ◄ primary" if metric == "ROC-AUC" else \
                 "  ◄ business priority" if metric == "Recall" else ""
        print(f"  {metric:<12}: {value:.4f}{marker}")
    print(f"{'='*40}\n")


# ── 6. Save model ────────────────────────────────────────────────────────────

def save_model(model, model_name: str, folder: str = "../models"):
    """Save trained model as .pkl file."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{model_name}.pkl")
    joblib.dump(model, path)
    print(f"Model saved to: {path}")
    return path


# ── 7. Load model ────────────────────────────────────────────────────────────

def load_model(model_name: str, folder: str = "../models"):
    """Load a saved model from .pkl file."""
    path = os.path.join(folder, f"{model_name}.pkl")
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    return model
