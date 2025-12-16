"""
app.py

Flask Web Application for ToxiDose AI.

Structure:
- Root ('/'):       Renders the Landing Page (base.html).
- Index ('/index'): Renders the Input Form (index.html).
- Prediction:
    - '/predict_manual' : Handles single-patient form submission.
    - '/predict_csv'    : Handles batch CSV upload.
- '/download_report'    : Generates PDF report from last single-patient run.
"""

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    make_response,
)
import pandas as pd
import sys
import os
import io
import base64
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt

from xhtml2pdf import pisa

# Ensure we can import from src (project root has src/)

# Ensure we can import from the project root (which contains 'src/')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from src.predict import RiskPredictor
from src.drug_table import get_drug_knowledge_base

app = Flask(__name__)
app.secret_key = "super_secret_hospital_key"  # Needed for flashing messages & session

# --------------------------------------------------
# MODEL + DRUG KB
# --------------------------------------------------
predictor = RiskPredictor()

drug_kb = get_drug_knowledge_base()
drug_names_list = sorted([d["drug_name"] for d in drug_kb])

# Optional: quick lookup by drug name
drug_lookup = {d["drug_name"]: d for d in drug_kb}


# --------------------------------------------------
# HELPER FUNCTIONS (for PDF charts & summaries)
# --------------------------------------------------
def _fig_to_base64(fig):
    """Convert a Matplotlib figure to a base64 PNG data URL."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def build_organ_radar(organ_loads: dict) -> str:
    """
    Builds a radar chart for organ stress profile.
    Expects keys: liver, kidney, heart, gi, lungs
    """
    labels = ["Liver", "Kidney", "Heart", "GI", "Lungs"]
    values = [
        float(organ_loads.get("liver", 0.0)),
        float(organ_loads.get("kidney", 0.0)),
        float(organ_loads.get("heart", 0.0)),
        float(organ_loads.get("gi", 0.0)),
        float(organ_loads.get("lungs", 0.0)),
    ]

    max_val = max(values) or 1.0
    norm = [v / max_val * 10.0 for v in values]

    # Radar setup
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    norm += norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, norm)
    ax.fill(angles, norm, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Organ Stress Profile", pad=20)

    return _fig_to_base64(fig)

def html_to_pdf(source_html: str) -> bytes:
    """
    Convert HTML string to PDF bytes using xhtml2pdf (pisa).
    Raises RuntimeError if PDF generation fails.
    """
    result = io.BytesIO()
    pisa_status = pisa.CreatePDF(src=source_html, dest=result)

    if pisa_status.err:
        raise RuntimeError("Error while generating PDF report")

    result.seek(0)
    return result.read()



def build_med_contrib_chart(med_contrib: list) -> str:
    """
    Horizontal bar chart of per-drug contribution to overall toxicity (%).
    Expects list of dicts: {'name': str, 'percent': float}
    """
    if not med_contrib:
        # Avoid empty plot
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No medicines in regimen", ha="center", va="center")
        ax.axis("off")
        return _fig_to_base64(fig)

    meds = [m["name"] for m in med_contrib]
    vals = [m["percent"] for m in med_contrib]

    fig, ax = plt.subplots()
    ax.barh(meds, vals)
    ax.invert_yaxis()
    ax.set_xlabel("Relative contribution to toxicity (%)")
    ax.set_title("Drug Contribution to Overall Risk")
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}%", va="center")

    return _fig_to_base64(fig)


def build_patient_summary_for_report(patient: dict) -> dict:
    """
    Compresses raw patient dict into a nicer summary for report.html.
    """
    age = patient.get("patient_age")
    age_group = patient.get("age_group", "").capitalize()
    weight = patient.get("weight_kg")
    sex = patient.get("sex", "unknown")

    conditions = []
    if patient.get("has_liver_disease"):
        conditions.append("Liver disease")
    if patient.get("has_kidney_disease"):
        conditions.append("Kidney disease")
    if patient.get("has_heart_disease"):
        conditions.append("Heart disease")
    if patient.get("has_gi_ulcer_or_gastritis"):
        conditions.append("GI ulcer / gastritis")
    if patient.get("has_asthma_or_copd"):
        conditions.append("Asthma / COPD")
    if patient.get("has_diabetes"):
        conditions.append("Diabetes")
    if patient.get("is_pregnant") and sex == "female":
        conditions.append("Pregnant")

    if not conditions:
        cond_label = "No major chronic conditions reported"
    else:
        cond_label = ", ".join(conditions)

    return {
        "age_label": f"{age} years ({age_group})" if age is not None else "N/A",
        "weight": weight,
        "sex": sex.capitalize(),
        "conditions_label": cond_label,
    }


def build_regimen_summary_for_report(regimen: list) -> list:
    """
    Converts regimen list into a simpler summary that report.html can render.
    """
    summary = []
    for med in regimen:
        try:
            dose_per_unit = float(med.get("dose_mg_per_unit", 0.0))
            units = float(med.get("units_per_dose", 0.0))
            dose_mg = round(dose_per_unit * units, 1)
        except Exception:
            dose_mg = 0.0

        freq = int(med.get("doses_per_day", 0))
        days_taken = int(med.get("days_taken_so_far", 0))
        hours_since = float(med.get("hours_since_last_dose", 0.0))

        summary.append(
            {
                "name": med.get("drug_name", "Unknown"),
                "dose_mg": dose_mg,
                "frequency_label": f"{freq}× / day",
                "days_taken": days_taken,
                "hours_since_last": hours_since,
            }
        )
    return summary


def build_med_contrib_from_regimen(regimen: list) -> list:
    """
    Roughly estimate per-drug toxicity contribution from dose and drug KB.
    This is for visualization only – not used for prediction.
    """
    contrib = []
    for med in regimen:
        name = med.get("drug_name", "Unknown")
        kb_row = drug_lookup.get(name)

        dose_per_unit = float(med.get("dose_mg_per_unit", 0.0))
        units = float(med.get("units_per_dose", 0.0))
        doses_per_day = float(med.get("doses_per_day", 0.0))

        daily_mg = dose_per_unit * units * doses_per_day

        if kb_row:
            max_daily = float(kb_row.get("max_daily_mg", 1.0)) or 1.0
            dose_ratio = daily_mg / max_daily

            tox_sum = (
                kb_row.get("tox_liver", 0)
                + kb_row.get("tox_kidney", 0)
                + kb_row.get("tox_heart", 0)
                + kb_row.get("tox_gi", 0)
                + kb_row.get("tox_lungs", 0)
            )
            tox_score = max(dose_ratio * tox_sum, 0.0)
        else:
            # Fallback: use daily mg as proxy
            tox_score = max(daily_mg, 0.0)

        contrib.append({"name": name, "toxicity": float(tox_score)})

    total = sum(c["toxicity"] for c in contrib) or 1.0
    for c in contrib:
        c["percent"] = round(c["toxicity"] / total * 100.0, 1)
    return contrib


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/")
def landing():
    """Renders the Landing / Home page (base.html)."""
    return render_template("base.html")


@app.route("/index")
def index():
    """Renders the Input Form page (index.html)."""
    return render_template("index.html", drug_names=drug_names_list)


@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    """Handles the single patient manual form submission."""
    try:
        # 1. Parse Patient Data
        age_val = int(request.form.get("age"))
        weight_val = float(request.form.get("weight"))

        patient = {
            "patient_age": age_val,
            "weight_kg": weight_val,
            "sex": request.form.get("sex"),
            "is_pregnant": 1 if request.form.get("is_pregnant") == "on" else 0,
            "has_liver_disease": 1 if request.form.get("liver_disease") == "on" else 0,
            "has_kidney_disease": 1 if request.form.get("kidney_disease") == "on" else 0,
            "has_heart_disease": 1 if request.form.get("heart_disease") == "on" else 0,
            "has_gi_ulcer_or_gastritis": 1 if request.form.get("gi_issue") == "on" else 0,
            "has_asthma_or_copd": 1 if request.form.get("asthma") == "on" else 0,
            "has_diabetes": 1 if request.form.get("diabetes") == "on" else 0,
            # Helper for feature engineering (must match training)
            "age_group": "child"
            if age_val < 18
            else ("elderly" if age_val >= 65 else "adult"),
        }

        # 2. Parse Regimen Data (Dynamic Rows)
        drug_names_input = request.form.getlist("drug_name[]")
        doses = request.form.getlist("dose[]")
        units = request.form.getlist("units[]")
        freqs = request.form.getlist("freq[]")
        days = request.form.getlist("days[]")
        hours = request.form.getlist("hours[]")

        regimen = []
        for i in range(len(drug_names_input)):
            if drug_names_input[i]:  # Skip empty rows
                regimen.append(
                    {
                        "drug_name": drug_names_input[i],
                        "dose_mg_per_unit": float(doses[i]),
                        "units_per_dose": int(units[i]),
                        "doses_per_day": int(freqs[i]),
                        "days_taken_so_far": int(days[i]),
                        "hours_since_last_dose": float(hours[i]),
                    }
                )

        if not regimen:
            flash("Please add at least one medicine.", "error")
            return redirect(url_for("index"))

        # 3. Check models
        if predictor.acute_model is None or predictor.cumul_model is None:
            flash(
                "Models not loaded. Please train models first (run models.py).",
                "error",
            )
            return redirect(url_for("index"))

        # 4. Get Prediction
        result = predictor.predict(patient, regimen)

        # 5. Generate SHAP Plot (Acute risk explanation)
        shap_plot = predictor.generate_explanation_plot(result["features"])

        # 6. Build structures needed for PDF report & store in session
        features = result.get("features", {})

        organ_loads = {
            "liver": float(features.get("effective_liver_load", 0.0)),
            "kidney": float(features.get("effective_kidney_load", 0.0)),
            "heart": float(features.get("effective_heart_load", 0.0)),
            "gi": float(features.get("effective_gi_load", 0.0)),
            "lungs": float(features.get("effective_lungs_load", 0.0)),
        }

        patient_summary = build_patient_summary_for_report(patient)
        regimen_summary = build_regimen_summary_for_report(regimen)
        med_contrib = build_med_contrib_from_regimen(regimen)

        # Save everything needed for /download_report
        session["last_report"] = {
            "acute_score": float(result.get("acute_risk_score", 0.0)),
            "acute_band": result.get("acute_bucket", "Unknown"),
            "cumulative_score": float(result.get("cumulative_risk_score", 0.0)),
            "cumulative_band": result.get("cumulative_bucket", "Unknown"),
            "patient": patient_summary,   # summarized for report
            "regimen": regimen_summary,   # simplified regimen
            "organ_loads": organ_loads,
            "med_contrib": med_contrib,
        }

        # 7. Render Result Page (single-patient view)
        return render_template(
            "result.html",
            mode="single",
            data=result,
            patient=patient,
            regimen=regimen,
            shap_plot=shap_plot,
        )

    except Exception as e:
        flash(f"Error processing manual input: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    """Handles CSV file upload for batch processing with RAW-like inputs."""
    if "file" not in request.files:
        flash("No file part", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file", "error")
        return redirect(url_for("index"))

    if file:
        try:
            # 1. Read CSV into DataFrame
            df = pd.read_csv(file)

            # Basic validation
            required_patient_cols = [
                "patient_age",
                "weight_kg",
                "sex",
                "is_pregnant",
                "has_liver_disease",
                "has_kidney_disease",
                "has_heart_disease",
                "has_gi_ulcer_or_gastritis",
                "has_asthma_or_copd",
                "has_diabetes",
            ]
            missing = [c for c in required_patient_cols if c not in df.columns]
            if missing:
                flash(f"Missing patient columns in CSV: {missing}", "error")
                return redirect(url_for("index"))

            # Check models
            if predictor.acute_model is None or predictor.cumul_model is None:
                flash(
                    "Models not loaded. Please train models first (run models.py).",
                    "error",
                )
                return redirect(url_for("index"))

            # Prepare output columns
            acute_preds = []
            cumul_preds = []

            # 2. For each row, build patient + regimen and call predictor
            max_meds = 5  # how many med slots we support in CSV
            for _, row in df.iterrows():
                # ----- Patient dict -----
                age_val = int(row["patient_age"])
                patient = {
                    "patient_age": age_val,
                    "weight_kg": float(row["weight_kg"]),
                    "sex": row["sex"],
                    "is_pregnant": int(row["is_pregnant"]),
                    "has_liver_disease": int(row["has_liver_disease"]),
                    "has_kidney_disease": int(row["has_kidney_disease"]),
                    "has_heart_disease": int(row["has_heart_disease"]),
                    "has_gi_ulcer_or_gastritis": int(
                        row["has_gi_ulcer_or_gastritis"]
                    ),
                    "has_asthma_or_copd": int(row["has_asthma_or_copd"]),
                    "has_diabetes": int(row["has_diabetes"]),
                    "age_group": "child"
                    if age_val < 18
                    else ("elderly" if age_val >= 65 else "adult"),
                }

                # ----- Regimen from repeated columns -----
                regimen = []
                for i in range(1, max_meds + 1):
                    name_col = f"drug{i}_name"
                    dose_col = f"drug{i}_dose_mg_per_unit"
                    units_col = f"drug{i}_units_per_dose"
                    freq_col = f"drug{i}_doses_per_day"
                    days_col = f"drug{i}_days_taken_so_far"
                    hours_col = f"drug{i}_hours_since_last_dose"

                    if name_col not in df.columns:
                        continue
                    if pd.isna(row[name_col]) or str(row[name_col]).strip() == "":
                        continue

                    regimen.append(
                        {
                            "drug_name": str(row[name_col]),
                            "dose_mg_per_unit": float(row.get(dose_col, 0.0)),
                            "units_per_dose": int(row.get(units_col, 1)),
                            "doses_per_day": int(row.get(freq_col, 1)),
                            "days_taken_so_far": int(row.get(days_col, 1)),
                            "hours_since_last_dose": float(row.get(hours_col, 6.0)),
                        }
                    )

                if not regimen:
                    # If no meds parsed, just skip / mark as 0 risk
                    acute_preds.append(0.0)
                    cumul_preds.append(0.0)
                    continue

                # Predict for this row
                res = predictor.predict(patient, regimen)
                acute_preds.append(res.get("acute_risk_score", 0.0))
                cumul_preds.append(res.get("cumulative_risk_score", 0.0))

            # Attach to DF
            df["model_acute_risk_score"] = acute_preds
            df["model_cumulative_risk_score"] = cumul_preds

            # Convert to HTML table for results page
            table_html = df.to_html(classes="table table-striped", index=False)

            return render_template(
                "result.html",
                mode="batch",
                table_html=table_html,
            )

        except Exception as e:
            flash(f"Error processing CSV file: {str(e)}", "error")
            return redirect(url_for("index"))


@app.route("/download_report")
def download_report():
    data = session.get("last_report")
    if not data:
        flash("No recent analysis found to generate a report.", "error")
        return redirect(url_for("index"))

    radar_img = build_organ_radar(data["organ_loads"])
    contrib_img = build_med_contrib_chart(data["med_contrib"])

    html = render_template(
        "report.html",
        data=data,
        radar_img=radar_img,
        contrib_img=contrib_img,
    )

    # Use xhtml2pdf instead of WeasyPrint
    try:
        pdf_bytes = html_to_pdf(html)
    except RuntimeError as e:
        flash(str(e), "error")
        return redirect(url_for("index"))

    response = make_response(pdf_bytes)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=ToxiDose_Report.pdf"
    return response



@app.route("/how_it_works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/help")
def help_page():
    return render_template("help.html")


if __name__ == "__main__":
    app.run(debug=True)
