"""
feature_engineering.py

Core logic to transform raw Patient + Regimen data into a flat feature vector.

Pipeline:
1. Aggregated Regimen Stats (total dose, duration, recency).
2. Dose Ratios (daily vs max safe, cumulative vs max safe cumulative).
3. Raw Organ Loads (sum of drug toxicity * dose ratio).
4. Patient Vulnerability Factors (multipliers based on patient health).
5. Effective Organ Loads (Raw Load * Vulnerability).
6. Interaction Flags (NSAID combos, sedative combos, etc.).
7. Risk Flags (overdose / overload indicators).

This class is used BOTH:
- When generating the synthetic dataset (training).
- At inference time in RiskPredictor (prediction).

So any change here must be followed by regenerating the dataset + retraining.
"""

import numpy as np
from src.drug_table import get_drug_knowledge_base


class FeatureEngineer:
    def __init__(self):
        # Create a lookup dictionary for faster access by drug_name
        self.drug_kb = {d["drug_name"]: d for d in get_drug_knowledge_base()}

    # --------------------------------------------------
    # VULNERABILITY FACTORS
    # --------------------------------------------------
    def _get_vulnerability_factors(self, p):
        """
        Computes patient-specific vulnerability multipliers for each organ.
        Base factor is 1.0. Higher means more vulnerable.
        """

        age_group = p.get('age_group', 'adult')
        weight = p.get('weight_kg', 70.0)
        is_preg = p.get('is_pregnant', 0) == 1

        # --- LIVER ---
        v_liver = 1.0
        if p['has_liver_disease']:
            v_liver += 2.0
        if age_group == 'elderly':
            v_liver += 0.5
        if is_preg:
            v_liver += 0.5  # pregnancy load
        if age_group == 'child':
            v_liver += 0.3  # smaller reserve
        if p['has_diabetes']:
            v_liver += 0.2

        # --- KIDNEY ---
        v_kidney = 1.0
        if p['has_kidney_disease']:
            v_kidney += 2.5
        if age_group == 'elderly':
            v_kidney += 0.8  # GFR drops with age
        if p['has_diabetes']:
            v_kidney += 0.5
        if weight < 50:
            v_kidney += 0.3  # low body mass
        if age_group == 'child':
            v_kidney += 0.4  # paediatric caution

        # --- HEART ---
        v_heart = 1.0
        if p['has_heart_disease']:
            v_heart += 2.0
        if age_group == 'elderly':
            v_heart += 0.5
        if weight > 100:
            v_heart += 0.3  # obesity strain
        if age_group == 'child' and weight < 35:
            v_heart += 0.2

        # --- GI (Stomach) ---
        v_gi = 1.0
        if p['has_gi_ulcer_or_gastritis']:
            v_gi += 2.5
        if age_group == 'elderly':
            v_gi += 0.4
        if is_preg:
            v_gi += 0.4  # reflux, nausea
        if age_group == 'child':
            v_gi += 0.2

        # --- LUNGS ---
        v_lungs = 1.0
        if p['has_asthma_or_copd']:
            v_lungs += 2.0
        if age_group == 'child':
            v_lungs += 0.3  # smaller airways

        return v_liver, v_kidney, v_heart, v_gi, v_lungs


    # --------------------------------------------------
    # MAIN FEATURE COMPUTATION
    # --------------------------------------------------
    def compute_features(self, patient: dict, regimen: list[dict]) -> dict:
        """
        Args:
            patient (dict): Patient profile.
            regimen (list of dict): List of medicines taken.

        Returns:
            dict: Flat dictionary of all engineered features.
        """
        # Start with patient features
        features = patient.copy()

        # --- 1. Regimen Aggregates ---
        num_medicines = len(regimen)

        total_daily_dose = 0.0
        days_taken_list = []
        hours_since_list = []

        # Dose ratio lists
        daily_ratios = []
        cumulative_ratios = []

        # Organ Load Lists - Daily
        liver_loads_d = []
        kidney_loads_d = []
        heart_loads_d = []
        gi_loads_d = []
        lungs_loads_d = []

        # Organ Load Lists - Cumulative
        liver_loads_c = []
        kidney_loads_c = []
        heart_loads_c = []
        gi_loads_c = []
        lungs_loads_c = []

        # Counters for interaction flags
        cnt_liver_tox = 0
        cnt_kidney_tox = 0
        cnt_gi_tox = 0
        cnt_nsaid = 0
        cnt_paracetamol = 0
        cnt_sedative = 0
        cnt_antihist = 0

        for med in regimen:
            name = med["drug_name"]
            drug_info = self.drug_kb.get(name, {})

            # Basic stats
            daily_dose = (
                med["dose_mg_per_unit"]
                * med["units_per_dose"]
                * med["doses_per_day"]
            )
            total_daily_dose += daily_dose

            days_taken_list.append(med["days_taken_so_far"])
            hours_since_list.append(med["hours_since_last_dose"])

            # --- Ratios ---
            # Daily Ratio = Current Daily Dose / Max Safe Daily Dose
            max_daily = drug_info.get("max_daily_mg", 10000.0)
            d_ratio = daily_dose / max_daily
            daily_ratios.append(d_ratio)

            # Cumulative Ratio = (Daily Dose * Days) / (Max Daily * Safe Days)
            # This represents "How much of the total safe course have I consumed?"
            safe_days = drug_info.get("safe_days", 30.0)
            cumulative_dose = daily_dose * med["days_taken_so_far"]
            max_safe_cumulative = max_daily * safe_days
            c_ratio = cumulative_dose / max_safe_cumulative
            cumulative_ratios.append(c_ratio)

            # --- Organ Loads with scenario-specific factors ---
            # Base load = Toxicity Weight * Dose Ratio
            # Then we modulate by patient scenarios (Option A)

            liver_factor = 1.0
            kidney_factor = 1.0
            heart_factor = 1.0
            gi_factor = 1.0
            lungs_factor = 1.0

            # 1) Pregnancy + NSAID → GI more sensitive
            if patient.get('is_pregnant', 0) == 1 and drug_info.get('is_nsaid', 0) == 1:
                gi_factor += 0.25

            # 2) Pregnancy + high liver-tox drug
            if patient.get('is_pregnant', 0) == 1 and drug_info.get('tox_liver', 0) >= 2:
                liver_factor += 0.2

            # 3) Child + low weight → kidney vulnerability
            if patient.get('age_group') == 'child' and patient.get('weight_kg', 70.0) < 40:
                kidney_factor += 0.2

            # 4) Asthma/COPD + NSAID → lungs hit
            if patient.get('has_asthma_or_copd', 0) == 1 and drug_info.get('is_nsaid', 0) == 1:
                lungs_factor += 0.3

            # 5) GI ulcer + NSAID → GI hit
            if patient.get('has_gi_ulcer_or_gastritis', 0) == 1 and drug_info.get('is_nsaid', 0) == 1:
                gi_factor += 0.3

            # Daily Loads
            liver_loads_d.append(d_ratio * drug_info['tox_liver']  * liver_factor)
            kidney_loads_d.append(d_ratio * drug_info['tox_kidney'] * kidney_factor)
            heart_loads_d.append(d_ratio * drug_info['tox_heart']   * heart_factor)
            gi_loads_d.append(d_ratio * drug_info['tox_gi']         * gi_factor)
            lungs_loads_d.append(d_ratio * drug_info['tox_lungs']   * lungs_factor)

            # Cumulative Loads (Chronic stress)
            liver_loads_c.append(c_ratio * drug_info['tox_liver']   * liver_factor)
            kidney_loads_c.append(c_ratio * drug_info['tox_kidney'] * kidney_factor)
            heart_loads_c.append(c_ratio * drug_info['tox_heart']   * heart_factor)
            gi_loads_c.append(c_ratio * drug_info['tox_gi']         * gi_factor)
            lungs_loads_c.append(c_ratio * drug_info['tox_lungs']   * lungs_factor)


            # --- Counters for interactions ---
            if drug_info["tox_liver"] > 0:
                cnt_liver_tox += 1
            if drug_info["tox_kidney"] > 0:
                cnt_kidney_tox += 1
            if drug_info["tox_gi"] >= 2:
                cnt_gi_tox += 1  # Only count moderate/high GI irritants

            if drug_info["is_nsaid"]:
                cnt_nsaid += 1
            if drug_info["is_paracetamol_source"]:
                cnt_paracetamol += 1
            if drug_info["is_sedative"]:
                cnt_sedative += 1
            if drug_info["is_antihistamine"]:
                cnt_antihist += 1

        # --- Aggregating Basic Regimen Features ---
        features["num_medicines"] = num_medicines
        features["total_daily_dose_mg"] = round(total_daily_dose, 2)

        features["max_days_taken_so_far"] = (
            max(days_taken_list) if days_taken_list else 0
        )
        features["avg_days_taken_so_far"] = (
            float(np.mean(days_taken_list)) if days_taken_list else 0.0
        )

        features["min_hours_since_last_dose"] = (
            min(hours_since_list) if hours_since_list else 0.0
        )
        features["max_hours_since_last_dose"] = (
            max(hours_since_list) if hours_since_list else 0.0
        )
        features["avg_hours_since_last_dose"] = (
            float(np.mean(hours_since_list)) if hours_since_list else 0.0
        )

        features["max_single_drug_daily_ratio"] = (
            max(daily_ratios) if daily_ratios else 0.0
        )
        features["avg_daily_dose_ratio"] = (
            float(np.mean(daily_ratios)) if daily_ratios else 0.0
        )
        features["sum_daily_dose_ratio"] = float(sum(daily_ratios)) if daily_ratios else 0.0

        features["max_single_drug_cumulative_ratio"] = (
            max(cumulative_ratios) if cumulative_ratios else 0.0
        )
        features["avg_cumulative_dose_ratio"] = (
            float(np.mean(cumulative_ratios)) if cumulative_ratios else 0.0
        )
        features["sum_cumulative_dose_ratio"] = (
            float(sum(cumulative_ratios)) if cumulative_ratios else 0.0
        )

        # --- Aggregating Organ Loads ---
        # Raw loads (sum of individual drug contributions)
        l_d = sum(liver_loads_d)
        k_d = sum(kidney_loads_d)
        h_d = sum(heart_loads_d)
        g_d = sum(gi_loads_d)
        lu_d = sum(lungs_loads_d)

        l_c = sum(liver_loads_c)
        k_c = sum(kidney_loads_c)
        h_c = sum(heart_loads_c)
        g_c = sum(gi_loads_c)
        lu_c = sum(lungs_loads_c)

        features.update(
            {
                "liver_load_daily": l_d,
                "kidney_load_daily": k_d,
                "heart_load_daily": h_d,
                "gi_load_daily": g_d,
                "lungs_load_daily": lu_d,
                "liver_load_cumulative": l_c,
                "kidney_load_cumulative": k_c,
                "heart_load_cumulative": h_c,
                "gi_load_cumulative": g_c,
                "lungs_load_cumulative": lu_c,
            }
        )

        # --- Vulnerability Adjustment ---
        v_liv, v_kid, v_hrt, v_gi, v_lng = self._get_vulnerability_factors(patient)

        features.update(
            {
                "effective_liver_load": l_d * v_liv,
                "effective_kidney_load": k_d * v_kid,
                "effective_heart_load": h_d * v_hrt,
                "effective_gi_load": g_d * v_gi,
                "effective_lungs_load": lu_d * v_lng,
                "effective_liver_load_cumulative": l_c * v_liv,
                "effective_kidney_load_cumulative": k_c * v_kid,
                "effective_heart_load_cumulative": h_c * v_hrt,
                "effective_gi_load_cumulative": g_c * v_gi,
                "effective_lungs_load_cumulative": lu_c * v_lng,
            }
        )

        # --- Interaction Flags ---
        features["num_liver_drugs"] = cnt_liver_tox
        features["num_kidney_drugs"] = cnt_kidney_tox
        features["num_gi_irritant_drugs"] = cnt_gi_tox

        features["has_multi_liver_drugs"] = int(cnt_liver_tox > 1)
        features["has_multi_kidney_drugs"] = int(cnt_kidney_tox > 1)
        features["has_multi_gi_irritants"] = int(cnt_gi_tox > 1)

        features["has_strong_nsaid_combo"] = int(cnt_nsaid >= 2)
        features["has_duplicated_paracetamol_sources"] = int(cnt_paracetamol >= 2)
        features["has_sedative_combo"] = int(
            cnt_sedative >= 1 and cnt_antihist >= 1
        )  # Specific interaction type

        # Risk Flags
        features["has_liver_overdose_risk"] = int(features["effective_liver_load"] > 3.0)
        features["has_kidney_overload_risk"] = int(features["effective_kidney_load"] > 3.0)

        return features


if __name__ == "__main__":
    # Simple sanity test
    from src.patient_simulator import PatientSimulator
    from src.regimen_simulator import RegimenSimulator

    p_sim = PatientSimulator()
    r_sim = RegimenSimulator()
    fe = FeatureEngineer()

    p = p_sim.generate_patient()
    r = r_sim.generate_regimen(patient_profile=p)

    f = fe.compute_features(p, r)
    print("Feature keys:", len(f), "features")
    print("Sample effective liver load:", f["effective_liver_load"])
