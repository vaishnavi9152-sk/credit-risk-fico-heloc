Credit Risk Prediction System (FICO HELOC)

A production-grade machine learning system that predicts credit default risk using FICO HELOC data, with cost-sensitive decisioning, explainability (SHAP), API deployment, CI, and Docker.

This project simulates how banks and fintech companies decide whether to approve, review, or reject a loan application.

What This System Does

Given a person’s credit profile, the system returns:

Probability of default

Risk tier: LOW / MEDIUM / HIGH

Final decision: APPROVE / MANUAL_REVIEW / REJECT

Top risk factors (why the model thinks this is risky)

Top protective factors (what helps this person)

All results are exposed through a FastAPI web service.

Machine Learning Pipeline

Raw Data → Preprocessing → XGBoost Model → Probability → Cost-Sensitive Decision → SHAP Explainability → API

Dataset

FICO HELOC dataset (OpenML ID: 45554)

Each row represents one borrower

Target variable:

Good → Loan repaid

Bad → Defaulted

This is a supervised binary classification problem.

Model

The system uses XGBoost, a state-of-the-art tree-based ensemble model widely used in financial risk modeling.

Why XGBoost

Handles non-linear relationships

Works well with missing values

Performs extremely well on tabular financial data

Model Performance (Test Set)
Metric	Value
ROC-AUC	~0.80
Brier Score	~0.18

This means the model is strong at ranking risky vs safe borrowers and produces high-quality probability estimates.

Cost-Sensitive Decision System

Banks do not optimize for accuracy — they optimize for financial loss.

False Negative (approve a defaulter) → high cost

False Positive (reject a good customer) → lower cost

We explicitly model this:

False-Negative cost = 2

False-Positive cost = 1

The system searches for probability thresholds that minimize financial loss, not classification error.

This produces two thresholds:

t_low

t_high

Risk Tier	Rule
LOW	probability < t_low
MEDIUM	t_low ≤ probability < t_high
HIGH	probability ≥ t_high
Explainability (SHAP)

Every prediction is fully explainable.

The API returns:

Top features increasing default risk

Top features reducing default risk

Exact feature values used

This is critical for:

Regulatory compliance

Loan officer trust

Model auditing and fairness

Example Output
{
  "prob_bad": 0.842,
  "risk_tier": "HIGH",
  "decision": "REJECT",
  "top_risk_factors": [
    { "feature": "ExternalRiskEstimate", "value": 55, "impact": 0.7267 },
    { "feature": "PercentTradesNeverDelq", "value": 83, "impact": 0.559 }
  ],
  "top_protective_factors": [
    { "feature": "AverageMInFile", "value": 84, "impact": -0.1886 }
  ]
}

API (FastAPI)
Start locally
uvicorn credit_risk.api:app --reload

Open
http://localhost:8000/docs

Endpoint
POST /predict

Request Payload
{
  "data": {
    "ExternalRiskEstimate": 55,
    "MSinceOldestTradeOpen": 144,
    "PercentTradesNeverDelq": 83
  }
}


The API returns risk tier, decision, and SHAP explanations.

Dockerized

The entire system is containerized.

Build
docker build -t credit-risk-fico-heloc .

Run
docker run -p 8000:8000 credit-risk-fico-heloc


This trains the model at build time and serves the API.

CI/CD

GitHub Actions automatically:

Installs dependencies

Trains the model

Runs unit tests

Runs API tests

This ensures the system never breaks when code changes.