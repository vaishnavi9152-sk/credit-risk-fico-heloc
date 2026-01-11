import joblib
import pandas as pd
from src.credit_risk.features import transform_with_imputer


def load_artifacts(path="artifacts.joblib"):
    artifacts = joblib.load(path)
    return artifacts["model"], artifacts["imputer"], artifacts["feature_names"]


def risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def decision_policy(tier: str) -> str:
    if tier == "LOW":
        return "APPROVE"
    elif tier == "MEDIUM":
        return "MANUAL_REVIEW"
    else:
        return "REJECT"


def predict_one(raw_input: dict):
    model, imputer, feature_names = load_artifacts()

    df = pd.DataFrame([raw_input])
    X = transform_with_imputer(df, imputer)

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]

    prob = float(model.predict_proba(X)[:, 1][0])
    tier = risk_tier(prob)
    action = decision_policy(tier)

    return {"prob_bad": round(prob, 4), "risk_tier": tier, "decision": action}
