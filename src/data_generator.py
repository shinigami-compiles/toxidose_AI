"""
data_generator.py

Orchestrates the creation of the full synthetic dataset.
1. Loops N times to generate Patient + Regimen samples.
2. Computes features using FeatureEngineer.
3. Calculates 'Ground Truth' risk scores (Acute & Cumulative) using hidden formulas.
4. Saves the labeled dataset to CSV.
"""

import pandas as pd
import numpy as np
import os
from src.patient_simulator import PatientSimulator
from src.regimen_simulator import RegimenSimulator
from src.feature_engineering import FeatureEngineer

# --- HIDDEN RISK FORMULAS ---
# These simulate the "biological reality" the ML model needs to discover.

def calculate_acute_risk(feat):
    """
    Acute Risk: Immediate danger from current dose intensity and organ stress.
    Driven by: Daily Loads, Overdose Flags, Interaction Multipliers.
    """
    # Base risk from organ loads (weighted by organ criticality)
    # Heart and Liver failures are often most acute.
    base_score = (
        (feat['effective_liver_load'] * 5.0) + 
        (feat['effective_kidney_load'] * 3.0) +
        (feat['effective_heart_load'] * 6.0) +
        (feat['effective_lungs_load'] * 4.0) +
        (feat['effective_gi_load'] * 1.5)
    )

    # Multiplier for dangerous combinations
    multiplier = 1.0
    if feat['has_liver_overdose_risk']: multiplier += 0.5
    if feat['has_kidney_overload_risk']: multiplier += 0.4
    if feat['has_strong_nsaid_combo']: multiplier += 0.3
    if feat['has_sedative_combo']: multiplier += 0.4 # CNS depression risk
    
    # Recent dose factor: Risk is higher if taken very recently (high peak plasma)
    recency_factor = 1.0
    if feat['min_hours_since_last_dose'] < 1.0:
        recency_factor = 1.2
    elif feat['min_hours_since_last_dose'] > 12.0:
        recency_factor = 0.8

    raw_score = base_score * multiplier * recency_factor
    
    # Add noise (biological variability)
    noise = np.random.normal(0, 2.0)
    final_score = raw_score + noise
    
    # Scaling to 0-100 roughly (clipping logic later)
    # Empirically, raw scores might range 0 to 50+, we scale up.
    return final_score * 1.5 

def calculate_cumulative_risk(feat):
    """
    Cumulative Risk: Long-term damage potential.
    Driven by: Cumulative Loads, Duration, Chronic Exposure.
    """
    # Base risk from cumulative loads
    base_score = (
        (feat['effective_liver_load_cumulative'] * 4.0) + 
        (feat['effective_kidney_load_cumulative'] * 5.0) + # Kidneys hate chronic stress
        (feat['effective_heart_load_cumulative'] * 3.0) +
        (feat['effective_gi_load_cumulative'] * 2.0)
    )
    
    # Duration penalty
    duration_penalty = 0
    if feat['max_days_taken_so_far'] > 30:
        duration_penalty += 5.0
    if feat['max_days_taken_so_far'] > 90:
        duration_penalty += 10.0

    multiplier = 1.0
    if feat['has_multi_liver_drugs']: multiplier += 0.2
    if feat['has_multi_kidney_drugs']: multiplier += 0.3
    
    raw_score = (base_score * multiplier) + duration_penalty
    
    # Noise
    noise = np.random.normal(0, 3.0)
    final_score = raw_score + noise
    
    return final_score * 1.2

# --- GENERATION LOGIC ---

def generate_dataset(num_samples=30000, output_path='data/synthetic_medicine_toxicity_dataset.csv'):
    print(f"Generating {num_samples} samples...")
    
    p_sim = PatientSimulator()
    r_sim = RegimenSimulator()
    fe = FeatureEngineer()
    
    data_rows = []
    
    for i in range(num_samples):
        if i % 1000 == 0:
            print(f"  ... processed {i} rows")
            
        # 1. Simulate Entities
        patient = p_sim.generate_patient()
        regimen = r_sim.generate_regimen(patient_profile=patient)
        
        # 2. Engineer Features
        features = fe.compute_features(patient, regimen)
        
        # 3. Compute Targets (Ground Truth)
        acute_score = calculate_acute_risk(features)
        cumul_score = calculate_cumulative_risk(features)
        
        # Clip to 0-100
        features['acute_risk_score'] = np.clip(acute_score, 0, 100)
        features['cumulative_risk_score'] = np.clip(cumul_score, 0, 100)
        
        # 4. Add Bucket Labels (Optional, for classification tasks later)
        features['acute_risk_bucket'] = 'high' if features['acute_risk_score'] > 60 else ('caution' if features['acute_risk_score'] > 30 else 'low')
        features['cumulative_risk_bucket'] = 'high' if features['cumulative_risk_score'] > 60 else ('caution' if features['cumulative_risk_score'] > 30 else 'low')

        data_rows.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(data_rows)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(df[['acute_risk_score', 'cumulative_risk_score']].describe())

if __name__ == "__main__":
    generate_dataset(num_samples=30000)