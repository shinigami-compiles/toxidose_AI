"""
predict.py

Inference utility for the Medicine Toxicity Risk system (ToxiDose AI).

- Loads trained models (Acute & Cumulative).
- Accepts raw patient + regimen data.
- Re-engineers features using the shared FeatureEngineer logic.
- Returns predicted risk scores and interpretation.
- Optionally generates a SHAP-based explanation plot for Acute risk.
"""

import os
import io
import base64
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.feature_engineering import FeatureEngineer

# Important: Switch matplotlib to non-interactive mode for web server stability
plt.switch_backend("Agg")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root (one level up from src/)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Models directory
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

ACUTE_MODEL_PATH = os.path.join(MODEL_DIR, "acute_model_v2.joblib")
CUMUL_MODEL_PATH = os.path.join(MODEL_DIR, "cumulative_model_v2.joblib")


class RiskPredictor:
    """
    Wrapper around the trained models + feature engineering logic.

    Usage:
        predictor = RiskPredictor()
        result = predictor.predict(patient_profile_dict, regimen_list_of_dicts)
    """

    def __init__(self):
        self.fe = FeatureEngineer()
        self.acute_model = None
        self.cumul_model = None

        # SHAP explainer (lazy-init)
        self.explainer = None

        self._load_models()

    # --------------------------------------------------
    # MODEL LOADING
    # --------------------------------------------------
    def _load_models(self):
        """Loads acute and cumulative models if they exist on disk."""
        if os.path.exists(ACUTE_MODEL_PATH):
            self.acute_model = joblib.load(ACUTE_MODEL_PATH)
        else:
            print(f"Warning: Acute model not found at {ACUTE_MODEL_PATH}")

        if os.path.exists(CUMUL_MODEL_PATH):
            self.cumul_model = joblib.load(CUMUL_MODEL_PATH)
        else:
            print(f"Warning: Cumulative model not found at {CUMUL_MODEL_PATH}")

    # --------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------
    def _preprocess_input(self, features_dict: dict) -> pd.DataFrame:
        """
        Transforms the raw feature dictionary into a DataFrame matching model training.

        Steps:
        1. Converts dict -> single-row DataFrame.
        2. Encodes categorical 'sex' into numeric 'sex_code' (0/1).
        3. Drops non-numeric/reference columns used only for labels or display.
        4. Keeps only numeric/bool columns (same style as in models.py).
        """
        # 0. Dict -> DataFrame
        df = pd.DataFrame([features_dict])

        # 1. Map 'sex' to 'sex_code' like in models.py
        #    (0 = male, 1 = female; anything else -> 0)
        if "sex" in df.columns:
            df["sex_code"] = df["sex"].map({"male": 0, "female": 1, "other": 0}).fillna(0)
        else:
            df["sex_code"] = 0

        # 2. Drop columns that were not used in training
        drop_cols = [
            "age_group",
            "sex",
            "acute_risk_score",
            "cumulative_risk_score",
            "acute_risk_bucket",
            "cumulative_risk_bucket",
        ]
        cols_to_keep = [c for c in df.columns if c not in drop_cols]
        df_clean = df[cols_to_keep]

        # 3. Keep only numeric / bool columns
        df_clean = df_clean.select_dtypes(include=["number", "bool"])

        return df_clean

    # --------------------------------------------------
    # SHAP EXPLANATIONS
    # --------------------------------------------------
    def _init_shap(self):
        """
        Lazy loader for SHAP explainer to avoid slow startup.

        Uses TreeExplainer on the acute Random Forest model.
        """
        if self.explainer is None and self.acute_model is not None:
            try:
                self.explainer = shap.TreeExplainer(self.acute_model)
            except Exception as e:
                print(f"SHAP Init Error: {e}")
                self.explainer = None

    def generate_explanation_plot(self, features_dict: dict) -> str | None:
        """
        Generates a SHAP Waterfall plot for the acute risk prediction and returns
        the image as a Base64-encoded PNG string.

        Returns:
            base64_str (str) or None if explanation could not be generated.
        """
        self._init_shap()
        if not self.explainer:
            return None

        try:
            # 1. Convert single patient dict to aligned DataFrame
            df_input = self._preprocess_input(features_dict)

            # 2. Calculate SHAP values
            shap_values = self.explainer(df_input)

            # 3. Create Waterfall Plot (first row only)
            plt.figure(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)

            # 4. Save to Memory Buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.close()

            return plot_data
        except Exception as e:
            print(f"Plotting Error: {e}")
            plt.close()
            return None

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------
    def predict(self, patient_profile: dict, regimen: list[dict]) -> dict:
        """
        Predicts acute and cumulative toxicity risks.

        Args:
            patient_profile (dict): Patient demographics/history.
            regimen (list of dict): List of medicines with dosing info.

        Returns:
            dict with keys:
                - 'acute_risk_score': float
                - 'cumulative_risk_score': float
                - 'acute_bucket': str  ("Low Risk" / "Caution" / "High Risk")
                - 'cumulative_bucket': str
                - 'features': dict (the raw engineered feature vector)
        """
        # 1. Generate Feature Vector using the shared FeatureEngineer
        raw_features = self.fe.compute_features(patient_profile, regimen)

        # 2. Prepare DataFrame for the models
        X_input = self._preprocess_input(raw_features)

        # 3. Default result
        result = {
            "acute_risk_score": 0.0,
            "cumulative_risk_score": 0.0,
            "acute_bucket": "unknown",
            "cumulative_bucket": "unknown",
            "features": raw_features,
        }

        # -----------------------------
        # ACUTE RISK PREDICTION
        # -----------------------------
        if self.acute_model is not None:
            try:
                # Align columns with what the acute model was trained on
                if hasattr(self.acute_model, "feature_names_in_"):
                    expected_cols = self.acute_model.feature_names_in_
                    X_model = X_input.reindex(columns=expected_cols, fill_value=0)
                else:
                    X_model = X_input

                pred = self.acute_model.predict(X_model)[0]
                result["acute_risk_score"] = round(float(pred), 2)
                result["acute_bucket"] = self._get_bucket(pred)
            except Exception as e:
                print(f"Prediction Error (Acute): {e}")

        # -----------------------------
        # CUMULATIVE RISK PREDICTION
        # -----------------------------
        if self.cumul_model is not None:
            try:
                # Align columns with what the cumulative model was trained on
                if hasattr(self.cumul_model, "feature_names_in_"):
                    expected_cols = self.cumul_model.feature_names_in_
                    X_model = X_input.reindex(columns=expected_cols, fill_value=0)
                else:
                    X_model = X_input

                pred = self.cumul_model.predict(X_model)[0]
                result["cumulative_risk_score"] = round(float(pred), 2)
                result["cumulative_bucket"] = self._get_bucket(pred)
            except Exception as e:
                print(f"Prediction Error (Cumulative): {e}")

        return result

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def _get_bucket(self, score: float) -> str:
        """Maps a continuous risk score into a discrete bucket."""
        if score > 60:
            return "Overdose ! Health at High Risk.."
        if score > 30:
            return "Caution ! Leaning towards Overdose.."
        return "Not Overdose, Low Risk"


if __name__ == "__main__":
    # Simple manual test with random synthetic data
    from src.patient_simulator import PatientSimulator
    from src.regimen_simulator import RegimenSimulator

    predictor = RiskPredictor()

    p_sim = PatientSimulator()
    r_sim = RegimenSimulator()

    patient = p_sim.generate_patient()
    regimen = r_sim.generate_regimen(patient_profile=patient)

    print("\n--- Test Prediction ---")
    print(f"Patient: {patient['sex']}, Age: {patient['patient_age']}")
    print(f"Meds: {[m['drug_name'] for m in regimen]}")

    prediction = predictor.predict(patient, regimen)
    print("Acute:", prediction["acute_risk_score"], prediction["acute_bucket"])
    print("Cumul:", prediction["cumulative_risk_score"], prediction["cumulative_bucket"])
