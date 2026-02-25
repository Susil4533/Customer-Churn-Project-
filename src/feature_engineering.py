"""
feature_engineering.py
-----------------------
Transforms processed churn data into model-ready features.

Design decisions:
- Binary categoricals (Yes/No) → Label encoded (0/1) — simple and interpretable
- Multi-class categoricals     → One-hot encoded — avoids false ordinal assumptions
- Contract type                → Ordinal encoded — has natural order (month < 1yr < 2yr)
- Payment method               → One-hot + grouped flag (hypothesis-driven)
- Numerical features           → StandardScaler (tenure, MonthlyCharges, TotalCharges)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── 1. Binary columns (Yes / No only) ───────────────────────────────────────

BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode strict Yes/No and Male/Female columns as 0/1."""
    df = df.copy()
    mapping = {"Yes": 1, "No": 0, "Female": 1, "Male": 0}
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


# ── 2. Contract type — ordinal (hypothesis H1 driven) ───────────────────────

CONTRACT_ORDER = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}


def encode_contract_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal-encode Contract column.
    Rationale: Contract has a natural commitment order.
    Higher value = longer contract = lower churn (confirmed by H1).
    """
    df = df.copy()
    df["Contract"] = df["Contract"].map(CONTRACT_ORDER)
    return df


# ── 3. Payment method — one-hot + grouped flag (hypothesis H2 driven) ───────

def encode_payment_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode PaymentMethod and add a derived binary feature:
    PaymentMethod_AutoPay = 1 if automated, 0 if not.

    Rationale: H2 showed electronic check drives high churn.
    The AutoPay flag captures the behavioral lock-in concept directly.
    """
    df = df.copy()

    # Grouped flag — directly tied to H2 hypothesis
    df["PaymentMethod_AutoPay"] = df["PaymentMethod"].apply(
        lambda x: 1 if "automatic" in str(x).lower() else 0
    )

    # One-hot encode the original column
    dummies = pd.get_dummies(df["PaymentMethod"], prefix="PaymentMethod", drop_first=False)
    dummies.columns = [c.replace(" ", "_").replace("(", "").replace(")", "") for c in dummies.columns]

    df = pd.concat([df.drop(columns=["PaymentMethod"]), dummies], axis=1)
    return df


# ── 4. Remaining multi-class categoricals — one-hot encoded ─────────────────

MULTICLASS_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def encode_multiclass_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode multi-class categorical columns.
    drop_first=True to avoid multicollinearity in logistic regression.
    """
    df = df.copy()
    existing = [c for c in MULTICLASS_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True)
    # Clean column names
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


# ── 5. Scale numerical features ──────────────────────────────────────────────

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def scale_numerical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    StandardScale tenure, MonthlyCharges, TotalCharges.

    Rationale: Logistic regression and distance-based models are sensitive
    to feature magnitude. Scaling ensures fair contribution from each feature.
    Returns both the transformed df and the fitted scaler (needed for inference).
    """
    df = df.copy()
    scaler = StandardScaler()
    existing = [c for c in NUMERICAL_COLS if c in df.columns]
    df[existing] = scaler.fit_transform(df[existing])
    return df, scaler


# ── 6. Full pipeline ─────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Full feature engineering pipeline.
    Returns model-ready DataFrame and fitted scaler.
    """
    df = encode_binary_columns(df)
    df = encode_contract_type(df)
    df = encode_payment_method(df)
    df = encode_multiclass_columns(df)
    df, scaler = scale_numerical_features(df)
    return df, scaler


# ── 7. Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_path = "data/processed_churn.csv"
    output_path = "data/featured_churn.csv"

    df_raw = pd.read_csv(input_path)
    df_featured, scaler = engineer_features(df_raw)

    df_featured.to_csv(output_path, index=False)

    print("Feature engineering complete.")
    print(f"Saved to: {output_path}")
    print(f"Shape: {df_featured.shape}")
    print(f"\nFinal columns ({len(df_featured.columns)}):")
    for col in df_featured.columns:
        print(f"  {col}")