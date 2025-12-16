```md
# ToxiDose AI 💊🧠  
**AI-Powered Medicine Toxicity Risk Assessment System**

ToxiDose AI is an end-to-end machine learning system that predicts **acute** and **cumulative toxicity risk** caused by medicine regimens, using patient-specific clinical factors and drug-level toxicity knowledge.  
It is designed as a **decision-support & educational system**, not a medical diagnosis tool.

---

## 🚀 Key Highlights

- 🔬 **Hybrid AI System** – Rule-based pharmacological logic + ML regression models  
- 🧠 **Dual Risk Prediction** – Acute (short-term) & Cumulative (long-term) toxicity  
- 🫀 **Organ-Wise Modeling** – Liver, Kidney, Heart, GI tract, Lungs  
- 📊 **Explainable AI (XAI)** – SHAP-based feature importance visualization  
- 🧪 **Synthetic Clinical Data** – Realistic patient & regimen simulation  
- 🌐 **Flask Web App** – Manual input & CSV batch analysis  
- 📦 **Deployment-Ready** – Pretrained models included for instant demo

---

## ⚠️ Disclaimer

> **ToxiDose AI is NOT a medical diagnosis tool.**  
> It is intended for **academic, educational, and decision-support demonstration purposes only**.  
> Predictions must **never replace professional medical judgment**.

---

## 🧩 System Architecture Overview

```

Patient Profile + Medicine Regimen
↓
Feature Engineering Engine
(Dose ratios, organ loads,
vulnerability adjustment,
interaction flags)
↓
ML Risk Prediction Models
├── Acute Toxicity Risk
└── Cumulative Toxicity Risk
↓
Explainability (SHAP)
↓
Web Interface

```

---

## 🧬 Feature Engineering Highlights

- **Patient Factors**
  - Age, weight, pregnancy
  - Liver, kidney, heart, GI, lung conditions
- **Regimen Factors**
  - Daily dose ratios
  - Cumulative exposure ratios
  - Duration & recency of intake
- **Organ Load Modeling**
  - Raw toxicity load per organ
  - Vulnerability-adjusted effective load
- **Drug Interaction Flags**
  - NSAID combinations
  - Sedative overlaps
  - Duplicate paracetamol sources

---

## 🧠 Machine Learning Models

- **Problem Type:** Regression
- **Targets:**
  - Acute Risk Score (0–100)
  - Cumulative Risk Score (0–100)
- **Models Used:**
  - Baseline: Linear Regression
  - Production: Random Forest Regressor
- **Explainability:** SHAP (TreeExplainer)

Risk Buckets:
- **Low Risk**: < 30
- **Caution**: 30 – 60
- **High Risk**: > 60

---

## 🗂 Project Structure

```

medicine_toxicity_ml/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── synthetic_medicine_toxicity_dataset_30k.csv
│
├── models/
│   ├── acute_model.joblib
│   └── cumulative_model.joblib
│
├── src/
│   ├── drug_table.py
│   ├── patient_simulator.py
│   ├── regimen_simulator.py
│   ├── feature_engineering.py
│   ├── data_generator.py
│   ├── models.py
│   └── predict.py
│
└── app/
├── app.py
├── templates/
└── static/

````

---

## 🛠 Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/ToxiDose-AI.git
cd ToxiDose-AI
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv toxidoxe
toxidoxe\Scripts\activate   # Windows
# source toxidoxe/bin/activate  # Linux/Mac
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### Option A: Run with Pretrained Models (Recommended)

```bash
python app/app.py
```

Then open:

```
http://127.0.0.1:5000
```

---

### Option B: Retrain Models (Optional)

```bash
python src/models.py
python app/app.py
```

---

## 📊 Application Features

### 🔹 Manual Assessment

* Enter patient profile
* Add multiple medicines dynamically
* Get acute & cumulative toxicity risk
* View explainable AI output

### 🔹 CSV Batch Processing

* Upload structured CSV
* Predict toxicity risk for multiple patients
* Preview results instantly

---

## 📁 Included Artifacts

This repository includes:

* ✅ **Synthetic dataset** (`data/*.csv`)
* ✅ **Pretrained ML models** (`models/*.joblib`)

These are included **for demonstration and deployment convenience**.
In production systems, models and datasets should be generated or loaded dynamically.

---

## 🔐 Data Ethics & Safety

* No real patient data used
* Entire dataset is synthetic
* No personal identifiers stored
* No medical decisions automated

---

## 🚀 Deployment Notes

* Works on Render / Railway / local servers
* Models load instantly (no cold training)
* Flask-based deployment friendly
* Can be containerized with Docker

---

## 🧠 Learning Outcomes

This project demonstrates:

* End-to-end ML pipeline design
* Feature engineering from domain logic
* Explainable AI integration
* Safe healthcare-adjacent system design
* Production-grade project structuring

---

## 👨‍💻 Author

**Harshal**
Final-Year AI & Data Science Engineering Student
Focus: Medical AI, ML Systems, Explainable AI

---

## ⭐ Acknowledgements

Inspired by:

* Pharmacovigilance systems
* Drug safety research
* Explainable AI in healthcare

---

## 📌 License

This project is released for **academic and educational use only**.

```
```
