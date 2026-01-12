import joblib
import pandas as pd

from src.credit_risk.features import transform_with_imputer
from src.credit_risk.explain import explain_one


def load_artifacts(path="artifacts.joblib"):
    artifacts = joblib.load(path)
    return artifacts


def risk_tier(prob: float, t_low: float, t_high: float) -> str:
    if prob >= t_high:
        return "HIGH"
    elif prob < t_low:
        return "LOW"
    else:
        return "MEDIUM"


def decision_policy(tier: str) -> str:
    if tier == "LOW":
        return "APPROVE"
    elif tier == "MEDIUM":
        return "MANUAL_REVIEW"
    else:
        return "REJECT"


def predict_one(raw_input: dict, top_k: int = 5):
    art = load_artifacts()

    model = art["model"]
    imputer = art["imputer"]
    feature_names = art["feature_names"]

    t_low = art["thresholds"]["t_low"]
    t_high = art["thresholds"]["t_high"]

    df = pd.DataFrame([raw_input])
    X = transform_with_imputer(df, imputer)

    # Align columns safely
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]

    prob = float(model.predict_proba(X)[:, 1][0])
    tier = risk_tier(prob, float(t_low), float(t_high))
    action = decision_policy(tier)

    # SHAP explanation (top reasons)
    top_risk, top_protect = explain_one(model, X.iloc[[0]], top_k=top_k)

    return {
        "prob_bad": round(prob, 4),
        "risk_tier": tier,
        "decision": action,
        "thresholds_used": {"t_low": round(float(t_low), 3), "t_high": round(float(t_high), 3)},
        "costs_used": art.get("costs", None),
        "top_risk_factors": top_risk,
        "top_protective_factors": top_protect,
    }
