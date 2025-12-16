import numpy as np
import pandas as pd
import random

# -----------------------------
# CONFIG
# -----------------------------
N_SAMPLES = 30000       # number of patient-regimen rows to generate
MAX_MEDICINES = 5       # max medicines per patient


# -----------------------------
# 1. DRUG–ORGAN KNOWLEDGE BASE
# -----------------------------
def build_drug_table():
    """
    Synthetic but realistic-ish drug table.
    All doses in mg per unit (tablet/capsule/5ml spoon etc.).
    tox_* are relative weights (0 = none, 1 low, 2 med, 3 high).
    """
    data = [
        # name,         class,       default_strength_mg, max_daily_mg, safe_days, tox_liver, tox_kidney, tox_heart, tox_gi, tox_lungs, is_nsaid, is_para_src, is_sedative, is_antihistamine
        ("Paracetamol", "analgesic", 500,                4000,         7,         3,         0,          0,         0,      0,         0,        1,           0,           0),
        ("Ibuprofen",   "nsaid",     200,                1200,         10,        1,         2,          1,         2,      0,         1,        0,           0,           0),
        ("Diclofenac",  "nsaid",     50,                 150,          10,        1,         3,          1,         3,      0,         1,        0,           0,           0),
        ("Aspirin",     "nsaid",     75,                 300,          10,        2,         2,          2,         3,      0,         1,        0,           0,           0),
        ("DXM",         "cough",     10,                 120,          7,         2,         0,          0,         0,      0,         0,        0,           1,           0),
        ("Codeine",     "cough",     10,                 60,           7,         2,         1,          1,         0,      1,         0,        0,           1,           0),
        ("Cetirizine",  "antihist",  10,                 20,           30,        1,         0,          0,         0,      0,         0,        0,           1,           1),
        ("Chlorpheniramine","antihist",4,                24,           14,        1,         0,          0,         0,      0,         0,        0,           1,           1),
        ("Omeprazole",  "ppi",       20,                 40,           30,        0,         0,          0,         0,      0,         0,        0,           0,           0),
        ("Ranitidine",  "h2block",   150,                300,          30,        0,         0,          0,         0,      0,         0,        0,           0,           0),
        ("Amoxicillin", "antibiotic",500,                3000,         14,        1,         1,          0,         0,      0,         0,        0,           0,           0),
        ("Azithromycin","antibiotic",500,                1500,         5,         2,         1,          1,         0,      0,         0,        0,           0,           0),
        ("Metformin",   "antidiab",  500,                2000,         365,       0,         2,          0,         0,      0,         0,        0,           0,           0),
        ("Salbutamol",  "bronchod",  4,                  16,           30,        0,         0,          1,         0,      2,         0,        0,           0,           0),
        ("Tramadol",    "opioid",    50,                 400,          7,         2,         1,          1,         1,      1,         0,        0,           1,           0),
    ]
    cols = [
        "drug_name", "drug_class", "default_strength_mg", "max_daily_mg", "safe_days",
        "tox_liver", "tox_kidney", "tox_heart", "tox_gi", "tox_lungs",
        "is_nsaid", "is_paracetamol_source", "is_sedative", "is_antihistamine"
    ]
    df = pd.DataFrame(data, columns=cols)
    # Paracetamol as para_src
    df.loc[df["drug_name"] == "Paracetamol", "is_paracetamol_source"] = 1
    return df

DRUG_TABLE = build_drug_table()


# -----------------------------
# 2. UTILITY FUNCTIONS
# -----------------------------
def sample_patient():
    """Sample a synthetic patient profile."""
    age = np.random.randint(5, 90)  # child to elderly
    if age < 18:
        age_group = "child"
    elif age < 60:
        age_group = "adult"
    else:
        age_group = "elderly"

    # Weight depends loosely on age
    if age_group == "child":
        weight = np.random.normal(30, 8)
    elif age_group == "adult":
        weight = np.random.normal(65, 12)
    else:
        weight = np.random.normal(60, 10)
    weight = float(max(15, min(weight, 130)))  # clamp

    sex = random.choice(["male", "female"])
    is_pregnant = 0
    if sex == "female" and 18 <= age <= 45:
        is_pregnant = np.random.binomial(1, 0.05)

    # Organ diseases: low prevalence but non-zero
    def prob_by_age(base, factor_elder=3.0):
        if age_group == "elderly":
            return min(0.8, base * factor_elder)
        return base

    has_liver = np.random.binomial(1, prob_by_age(0.05))
    has_kidney = np.random.binomial(1, prob_by_age(0.03))
    has_heart = np.random.binomial(1, prob_by_age(0.04))
    has_gi = np.random.binomial(1, prob_by_age(0.07))
    has_lungs = np.random.binomial(1, prob_by_age(0.06))
    has_diabetes = np.random.binomial(1, prob_by_age(0.1))

    return dict(
        patient_age=int(age),
        age_group=age_group,
        weight_kg=float(round(weight, 1)),
        sex=sex,
        is_pregnant=int(is_pregnant),
        has_liver_disease=int(has_liver),
        has_kidney_disease=int(has_kidney),
        has_heart_disease=int(has_heart),
        has_gi_ulcer_or_gastritis=int(has_gi),
        has_asthma_or_copd=int(has_lungs),
        has_diabetes=int(has_diabetes),
    )


def sample_regimen():
    """Sample a regimen: up to MAX_MEDICINES drugs, with doses & durations."""
    num_meds = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.15, 0.05])
    num_meds = min(num_meds, MAX_MEDICINES)

    chosen_drugs = DRUG_TABLE.sample(num_meds, replace=False).reset_index(drop=True)

    meds = []
    for _, row in chosen_drugs.iterrows():
        default = row["default_strength_mg"]
        # Allow +- 20% variation for strength
        strength = default * np.random.uniform(0.8, 1.2)

        # units per dose: 0.5,1,2 typical
        units_per_dose = np.random.choice([0.5, 1, 1, 1, 2])
        doses_per_day = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])

        # treatment length: antibiotics shorter, chronic meds longer
        if row["drug_class"] in ["antibiotic", "cough"]:
            days_taken = np.random.randint(1, 8)
        elif row["drug_class"] in ["nsaid", "analgesic"]:
            days_taken = np.random.randint(1, 14)
        else:
            days_taken = np.random.randint(3, 60)

        # last dose time: most between 0 and 12h
        hours_since_last = np.random.exponential(3.0)
        hours_since_last = float(max(0.1, min(hours_since_last, 24.0)))

        meds.append(dict(
            drug_name=row["drug_name"],
            drug_class=row["drug_class"],
            default_strength_mg=float(default),
            max_daily_mg=float(row["max_daily_mg"]),
            safe_days=float(row["safe_days"]),
            tox_liver=int(row["tox_liver"]),
            tox_kidney=int(row["tox_kidney"]),
            tox_heart=int(row["tox_heart"]),
            tox_gi=int(row["tox_gi"]),
            tox_lungs=int(row["tox_lungs"]),
            is_nsaid=int(row["is_nsaid"]),
            is_paracetamol_source=int(row["is_paracetamol_source"]),
            is_sedative=int(row["is_sedative"]),
            is_antihistamine=int(row["is_antihistamine"]),
            dose_per_unit_mg=float(strength),
            units_per_dose=float(units_per_dose),
            doses_per_day=int(doses_per_day),
            days_taken_so_far=int(days_taken),
            hours_since_last_dose=float(round(hours_since_last, 2)),
        ))
    return meds


def compute_vulnerability_factor(patient, organ):
    """
    Compute organ-specific vulnerability factor based on:
    - age group, disease flag, pregnancy, weight
    Base = 1, increased by risk factors.
    """
    age_group = patient["age_group"]
    weight = patient["weight_kg"]
    is_preg = patient["is_pregnant"]

    base = 1.0

    # --- LIVER ---
    if organ == "liver":
        if patient["has_liver_disease"]:
            base += 0.8
        if age_group == "elderly":
            base += 0.25
        # pregnancy: liver works harder
        if is_preg:
            base += 0.15
        # children: smaller reserve
        if age_group == "child":
            base += 0.15

    # --- KIDNEY ---
    elif organ == "kidney":
        if patient["has_kidney_disease"]:
            base += 0.8
        if age_group == "elderly":
            base += 0.3
        # low weight = less distribution volume
        if weight < 50:
            base += 0.2
        # children: special care with renally-cleared drugs
        if age_group == "child":
            base += 0.15

    # --- HEART ---
    elif organ == "heart":
        if patient["has_heart_disease"]:
            base += 0.8
        if age_group == "elderly":
            base += 0.3
        # very low weight child – additional fragility
        if age_group == "child" and weight < 35:
            base += 0.1

    # --- GI TRACT ---
    elif organ == "gi":
        if patient["has_gi_ulcer_or_gastritis"]:
            base += 0.8
        # pregnancy – reflux, nausea
        if is_preg:
            base += 0.15
        # kids – thinner mucosa
        if age_group == "child":
            base += 0.1

    # --- LUNGS ---
    elif organ == "lungs":
        if patient["has_asthma_or_copd"]:
            base += 0.8
        if age_group == "child":
            base += 0.25  # smaller airways

    # Diabetes slightly increases metabolic / vascular risk
    if patient["has_diabetes"]:
        if organ in ("liver", "kidney", "heart"):
            base += 0.1

    # Mild cap to avoid exploding factors
    return float(min(base, 3.0))



# -----------------------------
# 3. MAIN DATA GENERATION
# -----------------------------
rows = []

for _ in range(N_SAMPLES):
    patient = sample_patient()
    meds = sample_regimen()
    num_meds = len(meds)

    # Aggregate regimen features
    total_daily_dose_mg = 0.0
    max_days_taken_so_far = 0
    sum_days_taken_so_far = 0
    min_hours_since_last = float("inf")
    max_hours_since_last = 0.0
    sum_hours_since_last = 0.0

    daily_ratios = []
    cumulative_ratios = []

    liver_daily = kidney_daily = heart_daily = gi_daily = lungs_daily = 0.0
    liver_cum = kidney_cum = heart_cum = gi_cum = lungs_cum = 0.0

    num_liver_drugs = num_kidney_drugs = num_gi_irritants = 0
    num_nsaids = 0
    num_para_src = 0
    num_sedatives = 0
    num_antihist = 0

    for med in meds:
        dose_per_unit = med["dose_per_unit_mg"]
        units_per_dose = med["units_per_dose"]
        doses_per_day = med["doses_per_day"]
        days_taken = med["days_taken_so_far"]
        hours_last = med["hours_since_last_dose"]
        max_daily = med["max_daily_mg"]
        safe_days = med["safe_days"]

        daily_mg = dose_per_unit * units_per_dose * doses_per_day
        total_daily_dose_mg += daily_mg

        daily_ratio = daily_mg / max_daily
        daily_ratios.append(daily_ratio)

        cumulative_mg = daily_mg * days_taken
        safe_cum_mg = max_daily * safe_days
        cumulative_ratio = cumulative_mg / safe_cum_mg
        cumulative_ratios.append(cumulative_ratio)

        max_days_taken_so_far = max(max_days_taken_so_far, days_taken)
        sum_days_taken_so_far += days_taken

        min_hours_since_last = min(min_hours_since_last, hours_last)
        max_hours_since_last = max(max_hours_since_last, hours_last)
        sum_hours_since_last += hours_last

        # Organ loads
        # Organ loads with scenario-specific factors
        liver_factor = 1.0
        kidney_factor = 1.0
        heart_factor = 1.0
        gi_factor = 1.0
        lungs_factor = 1.0

        # --- Scenario tweaks (Option A) ---

        # 1) Pregnancy + NSAID → GI more sensitive
        if patient["is_pregnant"] and med["is_nsaid"]:
            gi_factor += 0.25

        # 2) Pregnancy + high liver-tox drug → liver more sensitive
        if patient["is_pregnant"] and med["tox_liver"] >= 2:
            liver_factor += 0.2

        # 3) Child + very low weight → kidneys more vulnerable overall
        if patient["age_group"] == "child" and patient["weight_kg"] < 40:
            kidney_factor += 0.2

        # 4) Asthma/COPD + NSAID → lungs flare
        if patient["has_asthma_or_copd"] and med["is_nsaid"]:
            lungs_factor += 0.3

        # 5) GI ulcer + NSAID → GI hit
        if patient["has_gi_ulcer_or_gastritis"] and med["is_nsaid"]:
            gi_factor += 0.3

        # Apply factors to daily loads
        liver_daily += daily_ratio * med["tox_liver"] * liver_factor
        kidney_daily += daily_ratio * med["tox_kidney"] * kidney_factor
        heart_daily  += daily_ratio * med["tox_heart"] * heart_factor
        gi_daily     += daily_ratio * med["tox_gi"] * gi_factor
        lungs_daily  += daily_ratio * med["tox_lungs"] * lungs_factor

        # Apply same factors to cumulative loads
        liver_cum += cumulative_ratio * med["tox_liver"] * liver_factor
        kidney_cum += cumulative_ratio * med["tox_kidney"] * kidney_factor
        heart_cum  += cumulative_ratio * med["tox_heart"] * heart_factor
        gi_cum     += cumulative_ratio * med["tox_gi"] * gi_factor
        lungs_cum  += cumulative_ratio * med["tox_lungs"] * lungs_factor


        # Counts for flags
        if med["tox_liver"] > 0:
            num_liver_drugs += 1
        if med["tox_kidney"] > 0:
            num_kidney_drugs += 1
        if med["tox_gi"] >= 2:
            num_gi_irritants += 1
        if med["is_nsaid"]:
            num_nsaids += 1
        if med["is_paracetamol_source"]:
            num_para_src += 1
        if med["is_sedative"]:
            num_sedatives += 1
        if med["is_antihistamine"]:
            num_antihist += 1

    avg_days_taken_so_far = sum_days_taken_so_far / num_meds
    avg_hours_since_last = sum_hours_since_last / num_meds

    max_single_drug_daily_ratio = max(daily_ratios)
    avg_daily_dose_ratio = sum(daily_ratios) / num_meds
    sum_daily_dose_ratio = sum(daily_ratios)

    max_single_drug_cumulative_ratio = max(cumulative_ratios)
    avg_cumulative_dose_ratio = sum(cumulative_ratios) / num_meds
    sum_cumulative_dose_ratio = sum(cumulative_ratios)

    # Vulnerability factors
    liver_vuln = compute_vulnerability_factor(patient, "liver")
    kidney_vuln = compute_vulnerability_factor(patient, "kidney")
    heart_vuln = compute_vulnerability_factor(patient, "heart")
    gi_vuln = compute_vulnerability_factor(patient, "gi")
    lungs_vuln = compute_vulnerability_factor(patient, "lungs")

    effective_liver_load = liver_daily * liver_vuln
    effective_kidney_load = kidney_daily * kidney_vuln
    effective_heart_load = heart_daily * heart_vuln
    effective_gi_load = gi_daily * gi_vuln
    effective_lungs_load = lungs_daily * lungs_vuln

    effective_liver_load_cum = liver_cum * liver_vuln
    effective_kidney_load_cum = kidney_cum * kidney_vuln
    effective_heart_load_cum = heart_cum * heart_vuln
    effective_gi_load_cum = gi_cum * gi_vuln
    effective_lungs_load_cum = lungs_cum * lungs_vuln

    # Combination flags
    has_multi_liver_drugs = int(num_liver_drugs > 1)
    has_multi_kidney_drugs = int(num_kidney_drugs > 1)
    has_multi_gi_irritants = int(num_gi_irritants > 1)
    has_strong_nsaid_combo = int(num_nsaids >= 2)
    has_duplicated_paracetamol_sources = int(num_para_src >= 2)
    has_sedative_combo = int(num_sedatives >= 1 and num_antihist >= 1)
    has_liver_overdose_risk = int(liver_daily > 1.2)  # >120% of safe load
    has_kidney_overload_risk = int(kidney_daily > 1.2)

    # Time-based sanity for infinities
    if min_hours_since_last == float("inf"):
        min_hours_since_last = 24.0

    # Raw risk scores (hidden formula) – later rescaled to 0–100
    # Acute emphasises daily effective loads
    acute_raw = (
        1.5 * effective_liver_load +
        1.3 * effective_kidney_load +
        1.0 * effective_heart_load +
        0.8 * effective_gi_load +
        0.5 * effective_lungs_load +
        0.5 * max_single_drug_daily_ratio +
        0.2 * sum_daily_dose_ratio
    )

    # Cumulative emphasises cumulative loads + duration
    # Cumulative emphasises cumulative loads + duration (slightly stronger)
    cumulative_raw = (
        2.2 * effective_liver_load_cum +
        2.0 * effective_kidney_load_cum +
        1.4 * effective_heart_load_cum +
        1.1 * effective_gi_load_cum +
        0.6 * effective_lungs_load_cum +
        0.35 * max_single_drug_cumulative_ratio +
        0.25 * sum_cumulative_dose_ratio +
        0.07 * max_days_taken_so_far
    )

    # Extra bump for nasty chronic combos
    if has_multi_liver_drugs:
        cumulative_raw += 1.5
    if has_multi_kidney_drugs:
        cumulative_raw += 2.0
    if has_multi_gi_irritants:
        cumulative_raw += 1.0


    # Add some noise to avoid perfect determinism
    acute_raw += np.random.normal(0, 0.5)
    cumulative_raw += np.random.normal(0, 1.0)

    # Build row
    row = {}
    # Patient features
    row.update(patient)

    # Regimen features
    row.update(dict(
        num_medicines=num_meds,
        total_daily_dose_mg=float(round(total_daily_dose_mg, 2)),
        max_days_taken_so_far=int(max_days_taken_so_far),
        avg_days_taken_so_far=float(round(avg_days_taken_so_far, 2)),
        min_hours_since_last_dose=float(round(min_hours_since_last, 2)),
        max_hours_since_last_dose=float(round(max_hours_since_last, 2)),
        avg_hours_since_last_dose=float(round(avg_hours_since_last, 2)),
        max_single_drug_daily_ratio=float(round(max_single_drug_daily_ratio, 3)),
        avg_daily_dose_ratio=float(round(avg_daily_dose_ratio, 3)),
        sum_daily_dose_ratio=float(round(sum_daily_dose_ratio, 3)),
        max_single_drug_cumulative_ratio=float(round(max_single_drug_cumulative_ratio, 3)),
        avg_cumulative_dose_ratio=float(round(avg_cumulative_dose_ratio, 3)),
        sum_cumulative_dose_ratio=float(round(sum_cumulative_dose_ratio, 3)),
        liver_load_daily=float(round(liver_daily, 3)),
        kidney_load_daily=float(round(kidney_daily, 3)),
        heart_load_daily=float(round(heart_daily, 3)),
        gi_load_daily=float(round(gi_daily, 3)),
        lungs_load_daily=float(round(lungs_daily, 3)),
        liver_load_cumulative=float(round(liver_cum, 3)),
        kidney_load_cumulative=float(round(kidney_cum, 3)),
        heart_load_cumulative=float(round(heart_cum, 3)),
        gi_load_cumulative=float(round(gi_cum, 3)),
        lungs_load_cumulative=float(round(lungs_cum, 3)),
        effective_liver_load=float(round(effective_liver_load, 3)),
        effective_kidney_load=float(round(effective_kidney_load, 3)),
        effective_heart_load=float(round(effective_heart_load, 3)),
        effective_gi_load=float(round(effective_gi_load, 3)),
        effective_lungs_load=float(round(effective_lungs_load, 3)),
        effective_liver_load_cumulative=float(round(effective_liver_load_cum, 3)),
        effective_kidney_load_cumulative=float(round(effective_kidney_load_cum, 3)),
        effective_heart_load_cumulative=float(round(effective_heart_load_cum, 3)),
        effective_gi_load_cumulative=float(round(effective_gi_load_cum, 3)),
        effective_lungs_load_cumulative=float(round(effective_lungs_load_cum, 3)),
        num_liver_drugs=int(num_liver_drugs),
        num_kidney_drugs=int(num_kidney_drugs),
        num_gi_irritant_drugs=int(num_gi_irritants),
        has_multi_liver_drugs=int(has_multi_liver_drugs),
        has_multi_kidney_drugs=int(has_multi_kidney_drugs),
        has_multi_gi_irritants=int(has_multi_gi_irritants),
        has_strong_nsaid_combo=int(has_strong_nsaid_combo),
        has_duplicated_paracetamol_sources=int(has_duplicated_paracetamol_sources),
        has_sedative_combo=int(has_sedative_combo),
        has_liver_overdose_risk=int(has_liver_overdose_risk),
        has_kidney_overload_risk=int(has_kidney_overload_risk),
        acute_raw=acute_raw,
        cumulative_raw=cumulative_raw,
    ))
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# -----------------------------
# 4. RESCALE RAW RISK TO 0–100
# -----------------------------
def rescale_to_0_100(series):
    mn = series.min()
    mx = series.max()
    if mx == mn:
        return pd.Series(np.zeros_like(series), index=series.index)
    return (series - mn) / (mx - mn) * 100.0

df["acute_risk_score"] = rescale_to_0_100(df["acute_raw"]).round(1)
df["cumulative_risk_score"] = rescale_to_0_100(df["cumulative_raw"]).round(1)

# Optional: create buckets for analysis (not needed for training if you only do regression)
df["acute_risk_bucket"] = pd.cut(
    df["acute_risk_score"],
    bins=[-0.1, 20, 50, 80, 100],
    labels=["low", "caution", "high", "very_high"]
)
df["cumulative_risk_bucket"] = pd.cut(
    df["cumulative_risk_score"],
    bins=[-0.1, 20, 50, 80, 100],
    labels=["low", "caution", "high", "very_high"]
)

# Drop raw internal fields if you want a clean dataset:
df_final = df.drop(columns=["acute_raw", "cumulative_raw"])

# -----------------------------
# 5. SAVE TO CSV
# -----------------------------
output_path = "synthetic_medicine_toxicity_dataset_30k_v2.csv"
df_final.to_csv(output_path, index=False)
print(f"Dataset generated with shape {df_final.shape} and saved to {output_path}")
print(df_final.head())
