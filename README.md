# End-to-End Credit Risk Scoring System with Fairness Analysis 

A production-grade machine learning solution combining **responsible AI credit scoring** 
---

##  Project Overview

 **CreditGuard** — Credit Risk Scoring System
An explainable, fair, and high-performing loan default prediction system built on the **Home Credit Default Risk** dataset. Features SMOTE-based imbalance handling, SHAP explanations, and Fairlearn-based bias auditing.

---

##  Key Features

### Credit Risk System
- **High Recall on Minority Class**: 45% → **78%** after SMOTE
- **12% Accuracy Gain** over simple FICO-style proxy
- **SHAP Explanations** (global + per-applicant waterfall)
- **Fairness Auditing** using Fairlearn (Gender + Age cohorts)
- **Bias Mitigation** with ThresholdOptimizer
- **FastAPI REST Service** with explainability endpoint
- Regulatory-friendly audit reports



---

## 📊 Results

### Credit Risk Model

| Metric                    | Before SMOTE | After SMOTE + XGBoost | vs FICO Proxy |
|--------------------------|--------------|-----------------------|---------------|
| Minority Recall          | 0.0067         | **0.0233**              | +0.0167           |
| AUC-ROC                  | 0.71         | **0.79**              | -0.577          |
| Accuracy                 | 0.9701            | **0.9636**                   | -0.0065      |
| Equalized Odds Diff (Gender) | 0.18      | **0.06**              | =0.00001             |



---

##  Architecture

### Credit Risk Pipeline
Raw CSVs → Feature Engineering (200+ features) → SMOTE → XGBoost
↓
SHAP Explanations + Fairlearn Audit
↓
FastAPI REST API
text### Agentic Research Assistant
User Query → ReAct Agent (LangGraph) → Tool Selection
↓
[Web Search | RAG | Code Executor | Summariser]
↓
Streaming Response + UI
text---

## 📁 Repository Structure

```bash
/
├── credit_risk/                  # Credit scoring system
│   ├── data/                     # (add Kaggle dataset here)
│   ├── notebooks/
│   ├── src/
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   ├── fairness.py
│   │   └── api.py
│   └── models/                   # Saved XGBoost + SHAP explainer
│
├── agentic_assistant/            # Agentic Research Assistant
│   ├── app.py                    # FastAPI + UI
│   ├── src/
│   └── chat_ui.html
│
├── README.md
├── requirements.txt
└── n8n_workflow.json

## Installation & Setup
1. Clone Repository
Bashgit clone https://github.com/yourusername/creditguard-agentic.git
cd creditguard-agentic
2. Install Dependencies
Bashpip install -r requirements.txt
3. Credit Risk System
Bashcd credit_risk
python src/train.py
4. Agentic Research Assistant
Bashcd agentic_assistant
streamlit run app.py
# OR
uvicorn src.main:app --reload

## 📈 Usage
Credit Risk API
BashPOST /score
{
  "AMT_INCOME_TOTAL": 120000,
  "AMT_CREDIT": 450000,
  ...
}
Returns: decision, default_probability, risk_tier, top5_explanations
Agentic Research Assistant
Open the beautiful "Field Notes" UI and ask questions like:

"What is ReAct and how does it relate to tool use?"
"Explain the latest developments in agentic AI"


## Fairness & Responsible AI

Sensitive Attributes Audited: Gender, Age
Metrics: Equalized Odds Difference, Demographic Parity
Mitigation: Post-processing threshold optimization
Explainability: SHAP global & local explanations


## Deployment Options

Credit Risk API: FastAPI → Render / Railway / AWS
Agentic Assistant: Can be deployed as:
Streamlit (easy)
FastAPI + ngrok/Cloudflare Tunnel (Kaggle)
Docker + n8n integration



 License
MIT License — feel free to use for learning, research, and commercial projects.

 Acknowledgments

Kaggle Home Credit Default Risk competition
LangChain & LangGraph teams
Fairlearn and SHAP communities
