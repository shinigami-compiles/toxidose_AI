"""
drug_table.py

Defines the static Knowledge Base for drugs.

Each entry contains:
- Identity:      name, class
- Dosage:        default unit strength, max daily dose, safe duration
- Toxicity:      organ toxicity weights (0–3) for liver, kidney, heart, GI, lungs
- Interactions:  flags such as NSAID, sedative, antihistamine, paracetamol source

This is used by:
- RegimenSimulator (to sample realistic regimens)
- FeatureEngineer  (to compute organ loads, ratios, interaction flags)
"""

from typing import List, Dict, Any
import pandas as pd


def get_drug_knowledge_base() -> List[Dict[str, Any]]:
    """
    Returns a list of dictionaries representing the drug database.

    NOTE:
    - Field names and semantics must NOT be changed without updating
      FeatureEngineer and RegimenSimulator accordingly.
    """
    drugs = [
        # --- ANALGESICS / NSAIDS ---
        {
            "drug_name": "Paracetamol",
            "drug_class": "Analgesic",
            "default_strength_mg": 500.0,
            "max_daily_mg": 4000.0,
            "safe_days": 10.0,
            "tox_liver": 3,  # High liver risk in overdose
            "tox_kidney": 1,
            "tox_heart": 0,
            "tox_gi": 0,
            "tox_lungs": 0,
            "is_nsaid": 0,
            "is_paracetamol_source": 1,
            "is_sedative": 0,
            "is_antihistamine": 0,
        },
        {
            "drug_name": "Ibuprofen",
            "drug_class": "NSAID",
            "default_strength_mg": 400.0,
            "max_daily_mg": 3200.0,
            "safe_days": 14.0,
            "tox_liver": 1,
            "tox_kidney": 2,  # Moderate kidney risk
            "tox_heart": 1,
            "tox_gi": 3,  # High GI risk
            "tox_lungs": 0,
            "is_nsaid": 1,
            "is_paracetamol_source": 0,
            "is_sedative": 0,
            "is_antihistamine": 0,
        },
        {
            "drug_name": "Diclofenac",
            "drug_class": "NSAID",
            "default_strength_mg": 50.0,
            "max_daily_mg": 150.0,
            "safe_days": 7.0,
            "tox_liver": 1,
            "tox_kidney": 2,
            "tox_heart": 2,  # Cardiovascular risk
            "tox_gi": 2,
            "tox_lungs": 0,
            "is_nsaid": 1,
            "is_paracetamol_source": 0,
            "is_sedative": 0,
            "is_antihistamine": 0,
        },
        {
            "drug_name": "Aspirin",
            "drug_class": "NSAID",
            "default_strength_mg": 300.0,
            "max_daily_mg": 4000.0,
            "safe_days": 30.0,  # Often taken long term but monitored
            "tox_liver": 0,
            "tox_kidney": 1,
            "tox_heart": 0,  # Actually protective, but high dose is toxic
            "tox_gi": 3,  # High GI ulcer risk
            "tox_lungs": 1,  # Can trigger asthma
            "is_nsaid": 1,
            "is_paracetamol_source": 0,
            "is_sedative": 0,
            "is_antihistamine": 0,
        },
        {
            "drug_name": "Tramadol",
            "drug_class": "Opioid",
            "default_strength_mg": 50.0,
            "max_daily_mg": 400.0,
            "safe_days": 5.0,
            "tox_liver": 1,
            "tox_kidney": 1,
            "tox_heart": 1,
            "tox_gi": 1,
            "tox_lungs": 2,  # Respiratory depression risk
            "is_nsaid": 0,
            "is_paracetamol_source": 0,
            "is_sedative": 1,
            "is_antihistamine": 0,
        },

        # --- ANTIBIOTICS ---
        {
            "drug_name": "Amoxicillin",
            "drug_class": "Antibiotic",
            "default_strength_mg": 500.0,
            "max_daily_mg": 3000.0,
            "safe_days": 10.0,
            "tox_liver": 1,
            "tox_kidney": 1,
            "tox_heart": 0,
            "tox_gi": 1,
            "tox_lungs": 0,
            "is_nsaid": 0,
            "is_paracetamol_source": 0,
            "is_sedative": 0,
            "is_antihistamine": 0,
        },
        {
            "drug_name": "Azithromycin",
            "drug_class": "Antibiotic",
            "default_strength_mg": 500.0,
            "max_daily_mg": 500.0,
            "safe_days": 5.0,
            "tox_liver": 2,  # Known liver impact
            "tox_kidney": 0,
            "tox_heart": 2,  # QT prolongation risk
            "tox_gi": 1,
            "tox_lungs": 0,
            "is_nsaid": 0,
            "is_paracetamol_source": 0,
            "is_sedative": 0,
            "is_antihistamine": 0,
        },

        # --- RESPIRATORY / ALLERGY ---
        {
            "drug_name": "Cetirizine",
            "drug_class": "Antihistamine",
            "default_strength_mg": 10.0,
            "max_daily_mg": 20.0,
            "safe_days": 30.0,
            "tox_liver": 0,
            "tox_kidney": 1,
            "tox_heart": 0,
            "tox_gi": 0,
            "tox_lungs": 0,
            "is_nsaid": 0,
            "is_paracetamol_source": 0,
            "is_sedative": 1,  # Mild sedative
            "is_antihistamine": 1,
        },
        {
            "drug_name": "Chlorpheniramine",
            "drug_class": "Antihistamine",
            "default_strength_mg": 4.0,
            "max_daily_mg": 24.0,
            "safe_days": 7.0,
            "tox_liver": 1,
            "tox_kidney": 0,
            "tox_heart": 0,
            "tox_gi": 0,
            "tox_lungs": 0,
            "is_nsaid": 0,
            "is_paracetamol_source": 0,
            "is_sedative": 1,  # Stronger sedative
            "is_antihistamine": 1,
        },
        {
            "drug_name": "Dextromethorphan",
            "drug_class": "Cough Suppressant",
            "default_strength_mg": 15.0,
            "max_daily_mg": 120.0,
            "safe_days": 7.0,
            "tox_liver": 1,
            "tox_kidney": 0,
            "tox_heart": 0,
            "tox_gi": 0,
            "tox_lungs": 1,
            "is_nsaid": 0,
            "is_paracetamol_source": 0,
            "is_sedative": 1,  # Can be sedative in high doses
            "is_antihistamine": 0,
        },
        {
            "drug_name": "Salbutamol",
            "drug_class": "Bronchodilator",
            "default_strength_mg": 4.0,
            "max_daily_mg": 32.0,
            "safe_days": 365.0,
            "tox_liver": 0,
            "tox_kidney": 0,
            "tox_heart": 2,  # Palpitations/Tachycardia
            "tox_gi": 0,
            "tox_lungs": 0,  # Treats lungs, but overdose stresses heart
            "is_nsaid": 0,
            "is_paracetamol_source": 0,
            "is_sedative": 0,
            "is_antihistamine": 0,
        },

        # --- GASTROINTESTINAL ---
        {
            "drug_name": "Omeprazole",
            "drug_class": "PPI",
            "default_strength_mg": 20.0,
            "max_daily_mg": 40.0,
            "safe_days": 60.0,
            "tox_liver": 0,
            "tox_kidney": 1,  # Long-term kidney issues possible
            "tox_heart": 0,
            "tox_gi": 0,
            "tox_lungs": 0,
            "is_nsaid": 0,
            "is_paracetamol_source": 0,
            "is_sedative": 0,
            "is_antihistamine": 0,
        },
        {
            "drug_name": "Ranitidine",
            "drug_class": "H2 Blocker",
            "default_strength_mg": 150.0,
            "max_daily_mg": 300.0,
            "safe_days": 30.0,
            "tox_liver": 1,
            "tox_kidney": 0,
            "tox_heart": 0,
            "tox_gi": 0,
            "tox_lungs": 0,
            "is_nsaid": 0,
            "is_paracetamol_source": 0,
            "is_sedative": 0,
            "is_antihistamine": 0,
        },

        # --- CHRONIC / METABOLIC ---
        {
            "drug_name": "Metformin",
            "drug_class": "Antidiabetic",
            "default_strength_mg": 500.0,
            "max_daily_mg": 2000.0,
            "safe_days": 365.0,
            "tox_liver": 0,
            "tox_kidney": 2,  # Lactic acidosis risk if kidney weak
            "tox_heart": 0,
            "tox_gi": 2,  # Common GI upset
            "tox_lungs": 0,
            "is_nsaid": 0,
            "is_paracetamol_source": 0,
            "is_sedative": 0,
            "is_antihistamine": 0,
        },
        {
            "drug_name": "Atorvastatin",
            "drug_class": "Statin",
            "default_strength_mg": 20.0,
            "max_daily_mg": 80.0,
            "safe_days": 365.0,
            "tox_liver": 2,  # Can elevate liver enzymes
            "tox_kidney": 0,
            "tox_heart": 0,
            "tox_gi": 0,
            "tox_lungs": 0,
            "is_nsaid": 0,
            "is_paracetamol_source": 0,
            "is_sedative": 0,
            "is_antihistamine": 0,
        },
    ]
    return drugs


def get_drug_df() -> pd.DataFrame:
    """
    Returns the drug knowledge base as a Pandas DataFrame.

    Useful for:
    - Exploratory analysis
    - Joining with other data
    - Debugging / inspection
    """
    return pd.DataFrame(get_drug_knowledge_base())
