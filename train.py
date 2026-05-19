"""
Customer Churn - Training Script
================================
Reproduces the preprocessing from Telco_Customer_Churn_(eda).ipynb and trains the
final Random Forest + SMOTEENN model from Customer_Churn_ML_model.ipynb.

Usage:
    python train.py

Outputs:
    churn_model.pkl       trained sklearn RandomForestClassifier
    feature_columns.pkl   feature column order the model expects
"""
import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "telco_churn.csv"
RANDOM_STATE = 100


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode the raw Telco CSV."""
    # Drop ID column
    df = df.drop(columns=["customerID"])

    # TotalCharges has blank strings; coerce to numeric then drop nulls
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(how="any").reset_index(drop=True)

    # Collapse "No internet service" / "No phone service" into "No"
    no_internet_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in no_internet_cols:
        df[col] = df[col].replace({"No internet service": "No"})
    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    # Binary encode the Yes/No columns
    binary_cols = [
        "Churn",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
    ]
    for col in binary_cols:
        df[col] = np.where(df[col] == "Yes", 1, 0)

    df["gender"] = np.where(df["gender"] == "Male", 1, 0)

    # One-hot encode the remaining categoricals
    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"])

    return df


def main() -> None:
    print(f"Loading {DATA_PATH}...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  raw shape: {df_raw.shape}")

    df = preprocess(df_raw)
    print(f"  after preprocessing: {df.shape}")

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)
    feature_columns = list(X.columns)

    # Split first, then resample only training data (avoids data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("\nApplying SMOTEENN to training set...")
    smote = SMOTEENN(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"  resampled training shape: {X_train_resampled.shape}")
    print(f"  class balance after resampling: {pd.Series(y_train_resampled).value_counts().to_dict()}")

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        random_state=RANDOM_STATE,
        max_depth=6,
        min_samples_leaf=8,
    )
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate on the original (not resampled) test set
    y_pred = model.predict(X_test)
    print("\n=== Evaluation on held-out test set ===")
    print(f"Accuracy: {model.score(X_test, y_test):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Save outputs
    joblib.dump(model, "churn_model.pkl")
    joblib.dump(feature_columns, "feature_columns.pkl")
    print("\nSaved:")
    print("  churn_model.pkl")
    print("  feature_columns.pkl")


if __name__ == "__main__":
    main()
