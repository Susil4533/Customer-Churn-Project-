import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load raw dataset"""
    return pd.read_csv(path)


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric and drop missing rows"""
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    return df


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map churn to binary"""
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove IDs or non-ML useful columns"""
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df


def preprocess_data(path: str) -> pd.DataFrame:
    """Full preprocessing pipeline"""
    df = load_data(path)
    df = clean_total_charges(df)
    df = map_target(df)
    df = drop_irrelevant_columns(df)
    return df


if __name__ == "__main__":
    df = preprocess_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Save cleaned dataset
    output_path = "data/processed_churn.csv"
    df.to_csv(output_path, index=False)

    print("Preprocessing complete.")
    print("Saved cleaned data to:", output_path)
    print("Shape:", df.shape)