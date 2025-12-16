"""
regimen_simulator.py

Generates a synthetic medicine regimen for a patient.

A regimen consists of 1 to 5 medicines taken concurrently.

For each medicine, we simulate:
- Strength per unit (mg)
- Units per dose (e.g., 1 or 2 tablets)
- Frequency (doses per day)
- Duration (days taken so far)
- Recency (hours since last dose)

The output structure is a list of dicts, each with keys:
    - drug_name
    - dose_mg_per_unit
    - units_per_dose
    - doses_per_day
    - days_taken_so_far
    - hours_since_last_dose

This matches what FeatureEngineer and the manual UI expect.
"""

import numpy as np
from src.drug_table import get_drug_knowledge_base


class RegimenSimulator:
    def __init__(self, seed: int = 42):
        # Dedicated RNG for reproducible sampling
        self.rng = np.random.default_rng(seed)
        self.drug_db = get_drug_knowledge_base()
        self.drug_names = [d["drug_name"] for d in self.drug_db]

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def generate_regimen(self, patient_profile: dict | None = None) -> list[dict]:
        """
        Generates a list of medicines (dict) for a single patient.

        Args:
            patient_profile (dict, optional):
                Currently not heavily used, but can be used to bias
                drug selection (e.g., diabetic patient takes Metformin).
                For now we mostly use random sampling to create diverse
                interaction risks.

        Returns:
            list[dict]: Each dict is one medicine entry.
        """
        # 1. Decide number of medicines (1 to 5)
        # Probabilities: Most people take 1–3, fewer take 4–5.
        num_meds = self.rng.choice(
            [1, 2, 3, 4, 5],
            p=[0.3, 0.3, 0.2, 0.1, 0.1],
        )

        # 2. Select drugs (sampling without replacement to avoid same drug twice)
        # Note: In reality, people might take the same active ingredient under
        # different brands, but here we assume unique actives per regimen.
        selected_indices = self.rng.choice(
            len(self.drug_db),
            size=num_meds,
            replace=False,
        )
        selected_drugs = [self.drug_db[i] for i in selected_indices]

        regimen: list[dict] = []

        for drug in selected_drugs:
            # 3. Simulate dosage

            # Strength: usually use default_strength_mg from the drug KB
            strength = drug["default_strength_mg"]

            # Units per dose: 1 is standard, 2 is common, 3 is rare/overdose
            units_per_dose = self.rng.choice(
                [1, 2, 3],
                p=[0.7, 0.25, 0.05],
            )

            # Doses per day: 1 to 4 times
            doses_per_day = self.rng.choice(
                [1, 2, 3, 4],
                p=[0.4, 0.3, 0.2, 0.1],
            )

            # 4. Simulate Duration (days taken so far)
            # Logic: Short course (antibiotics/pain) vs long course (chronic).
            # Generate relative to 'safe_days' to create realistic risk.
            safe_days = drug["safe_days"]

            # 80% of the time: within safe limit.
            # 20% of the time: exceeds safe limit (risk factor).
            if self.rng.random() < 0.8:
                days_taken = self.rng.integers(
                    1,
                    max(2, int(safe_days) + 1),
                )
            else:
                # Exceeding safe duration
                days_taken = self.rng.integers(
                    int(safe_days),
                    int(safe_days * 3) + 2,
                )

            # Cap realistic max days (e.g., 365 for chronic meds)
            days_taken = min(int(days_taken), 365)

            # 5. Simulate Recency (hours since last dose)
            # If taken 1/day, likely 0–24 hrs. If 4/day, likely 0–6 hrs.
            avg_interval = 24 / doses_per_day
            hours_since_last = self.rng.uniform(0.1, avg_interval * 1.5)
            hours_since_last = round(float(hours_since_last), 2)

            med_entry = {
                "drug_name": drug["drug_name"],
                "dose_mg_per_unit": float(strength),
                "units_per_dose": int(units_per_dose),
                "doses_per_day": int(doses_per_day),
                "days_taken_so_far": int(days_taken),
                "hours_since_last_dose": hours_since_last,
            }
            regimen.append(med_entry)

        return regimen


if __name__ == "__main__":
    # Test
    sim = RegimenSimulator()
    regimen = sim.generate_regimen()
    print("Sample Regimen:")
    for med in regimen:
        print(med)
