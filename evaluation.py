"""
evaluation.py

Comprehensive Model Evaluation & Analysis.
1. Loads the trained models and the dataset.
2. Calculates detailed regression metrics (MAE, RMSE, R2).
3. Generates and saves visual plots:
   - Actual vs. Predicted (Scatter)
   - Residual Distribution (Histogram)
   - Feature Importance (Bar Chart)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration
DATA_PATH = r'D:\my_stuff\Projects\new_projects\ToxiDose AI\synthetic_toxidose_dataset_30k_v2.csv'
MODEL_DIR = 'models'
PLOT_DIR = 'plotss'

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

def load_data():
    """Loads and preprocesses data exactly like training."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing (Must match models.py logic)
    df['sex_code'] = df['sex'].map({'male': 0, 'female': 1})
    
    # Select Features
    drop_cols = [
        'acute_risk_score', 'cumulative_risk_score', 
        'acute_risk_bucket', 'cumulative_risk_bucket',
        'age_group', 'sex' 
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y_acute = df['acute_risk_score']
    y_cumul = df['cumulative_risk_score']
    
    return X, y_acute, y_cumul

def plot_actual_vs_predicted(y_true, y_pred, title, filename):
    """Generates a scatter plot of Ground Truth vs Model Prediction."""
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    plt.scatter(y_true, y_pred, alpha=0.5, color='#3498db', edgecolor='k', s=20)
    
    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', lw=2, linestyle='--')
    
    plt.xlabel("Actual Risk Score")
    plt.ylabel("Predicted Risk Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()
    print(f"   Saved plot: {filename}")

def plot_residuals(y_true, y_pred, title, filename):
    """Plots the distribution of errors (residuals)."""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='#2ecc71', bins=30)
    plt.axvline(0, color='#e74c3c', linestyle='--')
    
    plt.xlabel("Residual (Error)")
    plt.title(f"Residual Distribution - {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()
    print(f"   Saved plot: {filename}")

def plot_feature_importance(model, feature_names, title, filename):
    """Visualizes which features contributed most to the decision."""
    if not hasattr(model, 'feature_importances_'):
        print(f"   Skipping feature importance for {title} (Linear Model)")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Top 15 Features
    top_n = 15
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances[top_indices], y=np.array(feature_names)[top_indices], palette='viridis')
    
    plt.title(f"Top {top_n} Feature Importance - {title}")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()
    print(f"   Saved plot: {filename}")

def evaluate_model(model_path, X_test, y_test, target_name):
    """Loads a model and runs full evaluation."""
    print(f"\n--- Evaluating {target_name} Model ---")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    # 1. Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R2:   {r2:.4f}")
    
    # 2. Plots
    plot_actual_vs_predicted(y_test, y_pred, 
                             f"Actual vs Predicted ({target_name})", 
                             f"{target_name.lower()}_prediction_scatter.png")
    
    plot_residuals(y_test, y_pred, 
                   target_name, 
                   f"{target_name.lower()}_residuals.png")
    
    plot_feature_importance(model, X_test.columns, 
                            target_name, 
                            f"{target_name.lower()}_feature_importance.png")

def main():
    print("Loading data for evaluation...")
    X, y_acute, y_cumul = load_data()
    
    # Ensure we evaluate on unseen data (Test Set)
    # Using same random_state=42 ensures we get the exact same split as training
    _, X_test, _, y_a_test, _, y_c_test = train_test_split(
        X, y_acute, y_cumul, test_size=0.2, random_state=42
    )
    
    # Evaluate Acute Model
    evaluate_model(os.path.join(MODEL_DIR, 'acute_model_v2.joblib'), 
                   X_test, y_a_test, "Acute_Risk")
    
    # Evaluate Cumulative Model
    evaluate_model(os.path.join(MODEL_DIR, 'cumulative_model_v2.joblib'), 
                   X_test, y_c_test, "Cumulative_Risk")
    
    print(f"\nAnalysis Complete. Check the '{PLOT_DIR}' folder for graphs.")

if __name__ == "__main__":
    main()