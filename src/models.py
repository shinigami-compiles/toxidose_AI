"""
models.py

Trains and evaluates ML models for toxicity risk prediction.
Targets:
1. Acute Risk Score (Regression)
2. Cumulative Risk Score (Regression)

Models:
- Baseline: Linear Regression
- Production: Random Forest Regressor

Outputs:
- Prints evaluation metrics (MAE, RMSE, R2)
- Saves trained models to disk for later inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# PATHS
# --------------------------------------------------
# IMPORTANT:
# Your generate_dataset.py script saves:
#   "synthetic_medicine_toxicity_dataset_30k.csv"
# in the project root, so we point directly to that.
# --------------------------------------------------
# PATHS (Render + Local Safe)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "synthetic_medicine_toxicity_dataset_30k_v2.csv"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

ACUTE_MODEL_PATH = os.path.join(MODEL_DIR, "acute_model_v2.joblib")
CUMUL_MODEL_PATH = os.path.join(MODEL_DIR, "cumulative_model_v2.joblib")


def train_models():
    """
    Load the synthetic dataset, prepare features/targets, train models,
    evaluate them, and save the production Random Forest models.
    """
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. "
            f"Run generate_dataset.py first to create the 30k synthetic dataset."
        )

    df = pd.read_csv(DATA_PATH)

    # 2. Prepare Features (X) and Targets (y)
    # Drop targets and non-numeric/metadata columns
    drop_cols = [
        "acute_risk_score",
        "cumulative_risk_score",
        "acute_risk_bucket",
        "cumulative_risk_bucket",
        "age_group",
        "sex",  # raw string, we will replace it with numeric sex_code
    ]

    # Map 'sex' to a numeric code as done in inference
    # (0 = male, 1 = female; others/NaN default to 0)
    df["sex_code"] = df["sex"].map({"male": 0, "female": 1}).fillna(0)

    # All columns except dropped ones and the original 'sex' string
    feature_cols = [c for c in df.columns if c not in drop_cols and c != "sex"]

    X = df[feature_cols]
    y_acute = df["acute_risk_score"]
    y_cumul = df["cumulative_risk_score"]

    print(f"Training on {X.shape[0]} samples with {X.shape[1]} features.")
    print("Feature columns used for training:")
    print(feature_cols)

    # 3. Train / Test Split
    # We use the same split for both targets for consistency.
    X_train, X_test, y_a_train, y_a_test, y_c_train, y_c_test = train_test_split(
        X,
        y_acute,
        y_cumul,
        test_size=0.2,
        random_state=42,
    )

    # 4. Train Acute Risk Models
    print("\n--- Training Acute Risk Models ---")

    # Baseline: Linear Regression
    lr_acute = LinearRegression()
    lr_acute.fit(X_train, y_a_train)
    evaluate_model(lr_acute, X_test, y_a_test, "Linear Regression (Acute)")

    # Production: Random Forest
    rf_acute = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_acute.fit(X_train, y_a_train)
    evaluate_model(rf_acute, X_test, y_a_test, "Random Forest (Acute)")

    # 5. Train Cumulative Risk Models
    print("\n--- Training Cumulative Risk Models ---")

    # Baseline: Linear Regression
    lr_cumul = LinearRegression()
    lr_cumul.fit(X_train, y_c_train)
    evaluate_model(lr_cumul, X_test, y_c_test, "Linear Regression (Cumulative)")

    # Production: Random Forest
    rf_cumul = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_cumul.fit(X_train, y_c_train)
    evaluate_model(rf_cumul, X_test, y_c_test, "Random Forest (Cumulative)")

    # 6. Save Best Models (Random Forests)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf_acute, ACUTE_MODEL_PATH)
    joblib.dump(rf_cumul, CUMUL_MODEL_PATH)

    print(f"\nModels saved to '{MODEL_DIR}/'")
    print(f"  Acute model    -> {ACUTE_MODEL_PATH}")
    print(f"  Cumulative model -> {CUMUL_MODEL_PATH}")


def evaluate_model(model, X_test, y_test, name: str):
    """
    Print MAE, RMSE, and R^2 for a given model on the test set.
    """
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"[{name}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")


if __name__ == "__main__":
    train_models()
