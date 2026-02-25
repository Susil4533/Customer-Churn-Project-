"""
evaluation.py
-------------
Shared utilities for final model comparison, business translation,
and evaluation reporting.

Used by 04_evaluation.ipynb.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_curve,
    confusion_matrix,
)


# ── 1. Load all saved models ─────────────────────────────────────────────────

def load_all_models(folder: str = "../models") -> dict:
    """Load all three saved .pkl models and return as a dictionary."""
    model_names = {
        "Logistic Regression": "logistic_regression",
        "Random Forest":       "random_forest",
        "Gradient Boosting":   "gradient_boosting",
    }
    models = {}
    for display_name, file_name in model_names.items():
        path = os.path.join(folder, f"{file_name}.pkl")
        models[display_name] = joblib.load(path)
        print(f"Loaded: {display_name}")
    return models


# ── 2. Build comparison metrics table ────────────────────────────────────────

def build_metrics_table(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and return a clean comparison DataFrame.
    Sorted by ROC-AUC descending.
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model":      name,
            "ROC-AUC":    round(roc_auc_score(y_test, y_prob), 4),
            "Recall":     round(recall_score(y_test, y_pred), 4),
            "Precision":  round(precision_score(y_test, y_pred), 4),
            "F1":         round(f1_score(y_test, y_pred), 4),
            "Accuracy":   round(accuracy_score(y_test, y_pred), 4),
        })
    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return df


# ── 3. Overlaid ROC curve ─────────────────────────────────────────────────────

def plot_all_roc_curves(models: dict, X_test, y_test):
    """Plot all three ROC curves on a single chart for direct comparison."""
    colors = {
        "Logistic Regression": "#2980b9",
        "Random Forest":       "#27ae60",
        "Gradient Boosting":   "#8e44ad",
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve Comparison — All Models", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()


# ── 4. Metrics bar chart ──────────────────────────────────────────────────────

def plot_metrics_comparison(metrics_df: pd.DataFrame):
    """Side-by-side bar chart comparing ROC-AUC and Recall across models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#2980b9", "#8e44ad", "#27ae60"]

    for i, metric in enumerate(["ROC-AUC", "Recall"]):
        axes[i].bar(metrics_df["Model"], metrics_df[metric], color=colors)
        for j, val in enumerate(metrics_df[metric]):
            axes[i].text(j, val + 0.005, f"{val:.4f}", ha="center", fontsize=10)
        axes[i].set_title(f"{metric} by Model", fontsize=12)
        axes[i].set_ylabel(metric)
        axes[i].set_ylim(0, 1.05)
        axes[i].tick_params(axis='x', rotation=10)

    plt.suptitle("Primary Evaluation Metrics Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# ── 5. Risk scoring — business translation ───────────────────────────────────

def build_risk_table(model, X_test, y_test, model_name: str = "Logistic Regression") -> pd.DataFrame:
    """
    Assign churn probability scores to each test customer.
    Rank by risk and assign tiers: High / Medium / Low.
    This is the bridge between model output and business action.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    risk_df = pd.DataFrame({
        "Churn_Probability": y_prob,
        "Actual_Churn":      y_test.values,
    }).sort_values("Churn_Probability", ascending=False).reset_index(drop=True)

    # Assign risk tiers
    risk_df["Risk_Tier"] = pd.cut(
        risk_df["Churn_Probability"],
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"]
    )

    return risk_df


def plot_risk_tier_summary(risk_df: pd.DataFrame):
    """Summarize churn rates and customer counts by risk tier."""
    summary = risk_df.groupby("Risk_Tier", observed=True).agg(
        Customers=("Actual_Churn", "count"),
        Actual_Churn_Rate=("Actual_Churn", "mean")
    ).reset_index()
    summary["Actual_Churn_Rate"] = (summary["Actual_Churn_Rate"] * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    axes[0].bar(summary["Risk_Tier"], summary["Customers"], color=colors)
    for i, val in enumerate(summary["Customers"]):
        axes[0].text(i, val + 5, str(val), ha="center", fontsize=10)
    axes[0].set_title("Customer Count by Risk Tier")
    axes[0].set_ylabel("Number of Customers")

    axes[1].bar(summary["Risk_Tier"], summary["Actual_Churn_Rate"], color=colors)
    for i, val in enumerate(summary["Actual_Churn_Rate"]):
        axes[1].text(i, val + 0.5, f"{val}%", ha="center", fontsize=10)
    axes[1].set_title("Actual Churn Rate by Risk Tier")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_ylim(0, 100)

    plt.suptitle("Risk Tier Analysis — Business Action Layer", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    print("\nRisk Tier Summary:")
    print(summary.to_string(index=False))
    return summary


# ── 6. Top 20% high risk customers ───────────────────────────────────────────

def top_risk_customers(risk_df: pd.DataFrame, top_pct: float = 0.20) -> pd.DataFrame:
    """
    Extract top X% highest-risk customers.
    Business use: prioritize retention offers for this group only.
    """
    n = int(len(risk_df) * top_pct)
    top_df = risk_df.head(n)
    actual_churn_rate = top_df["Actual_Churn"].mean() * 100

    print(f"Top {int(top_pct*100)}% highest-risk customers: {n} customers")
    print(f"Actual churn rate in this group: {actual_churn_rate:.1f}%")
    print(f"vs. overall churn rate: {risk_df['Actual_Churn'].mean()*100:.1f}%")
    return top_df
