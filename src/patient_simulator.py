"""
patient_simulator.py

Generates synthetic patient profiles with realistic demographics
and medical history.

Enforces constraints like:
- Pregnancy only possible for females of reproductive age.
- Chronic diseases (heart, diabetes, etc.) more common in older age groups.
- Weight correlated with age (children weigh less).
"""

import numpy as np
import pandas as pd


class PatientSimulator:
    def __init__(self, seed: int = 42):
        # Use a dedicated RNG for reproducible sampling
        self.rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------
    def _get_age_group(self, age: int) -> str:
        """
        Map numeric age to a coarse age_group string.
        """
        if age < 18:
            return "child"
        elif age >= 65:
            return "elderly"
        else:
            return "adult"

    def _sample_weight(self, age: int, sex: str) -> float:
        """
        Generates a realistic weight (kg) based on age and sex.
        """
        # Base weight logic
        if age < 10:
            # Child: roughly 5 kg base + ~3 kg/year
            mean_w = 5 + (age * 3)
            std_w = 3
        elif age < 18:
            # Teen
            mean_w = 35 + (age - 10) * 4
            std_w = 8
        else:
            # Adult
            mean_w = 75 if sex == "male" else 65
            std_w = 15

        weight = self.rng.normal(mean_w, std_w)
        # Hard clamp to avoid crazy values
        weight = max(5.0, weight)
        return round(weight, 1)

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def generate_patient(self) -> dict:
        """
        Generates a single dictionary representing one synthetic patient.

        Returns keys:
            - patient_age (int)
            - age_group (str: child/adult/elderly)
            - weight_kg (float)
            - sex (str: male/female)
            - is_pregnant (int 0/1)
            - has_liver_disease (int 0/1)
            - has_kidney_disease (int 0/1)
            - has_heart_disease (int 0/1)
            - has_gi_ulcer_or_gastritis (int 0/1)
            - has_asthma_or_copd (int 0/1)
            - has_diabetes (int 0/1)
        """
        # 1. Demographics
        age = int(self.rng.integers(5, 90))  # Age between 5 and 90
        sex = self.rng.choice(["male", "female"], p=[0.49, 0.51])
        age_group = self._get_age_group(age)
        weight_kg = self._sample_weight(age, sex)

        # 2. Pregnancy Status
        is_pregnant = 0
        if sex == "female" and 18 <= age <= 45:
            # 5% chance of being pregnant for women of reproductive age
            is_pregnant = self.rng.choice([0, 1], p=[0.95, 0.05])

        # 3. Disease Probabilities (correlated with age)
        # Base probabilities
        p_liver = 0.05
        p_kidney = 0.03
        p_heart = 0.05
        p_gi = 0.10
        p_lung = 0.08
        p_diabetes = 0.08

        # Adjust for age (elderly have higher risk, children lower risk)
        if age_group == "elderly":
            p_liver += 0.05
            p_kidney += 0.10
            p_heart += 0.20
            p_gi += 0.05
            p_lung += 0.05
            p_diabetes += 0.15
        elif age_group == "child":
            p_liver *= 0.1
            p_kidney *= 0.1
            p_heart *= 0.1
            p_gi *= 0.5
            p_lung *= 1.2  # Asthma common in kids
            p_diabetes *= 0.1

        # 4. Sample conditions for this patient
        has_liver_disease = self.rng.choice([0, 1], p=[1 - p_liver, p_liver])
        has_kidney_disease = self.rng.choice([0, 1], p=[1 - p_kidney, p_kidney])
        has_heart_disease = self.rng.choice([0, 1], p=[1 - p_heart, p_heart])
        has_gi_ulcer_or_gastritis = self.rng.choice(
            [0, 1],
            p=[1 - p_gi, p_gi],
        )
        has_asthma_or_copd = self.rng.choice(
            [0, 1],
            p=[1 - p_lung, p_lung],
        )
        has_diabetes = self.rng.choice(
            [0, 1],
            p=[1 - p_diabetes, p_diabetes],
        )

        return {
            "patient_age": age,
            "age_group": age_group,
            "weight_kg": weight_kg,
            "sex": sex,
            "is_pregnant": int(is_pregnant),
            "has_liver_disease": int(has_liver_disease),
            "has_kidney_disease": int(has_kidney_disease),
            "has_heart_disease": int(has_heart_disease),
            "has_gi_ulcer_or_gastritis": int(has_gi_ulcer_or_gastritis),
            "has_asthma_or_copd": int(has_asthma_or_copd),
            "has_diabetes": int(has_diabetes),
        }

    def generate_batch(self, n: int = 1000) -> pd.DataFrame:
        """
        Generates a DataFrame of n patients.
        """
        patients = [self.generate_patient() for _ in range(n)]
        return pd.DataFrame(patients)


if __name__ == "__main__":
    # Quick test
    sim = PatientSimulator()
    df = sim.generate_batch(5)
    print(df)
