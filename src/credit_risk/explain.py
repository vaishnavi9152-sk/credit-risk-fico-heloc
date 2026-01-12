import pandas as pd
import shap

# Cache explainer so we don't rebuild it every request
_EXPLAINER = None
_MODEL_ID = None


def _get_tree_explainer(model):
    global _EXPLAINER, _MODEL_ID
    mid = id(model)
    if _EXPLAINER is None or _MODEL_ID != mid:
        _EXPLAINER = shap.TreeExplainer(model)
        _MODEL_ID = mid
    return _EXPLAINER


def explain_one(model, X_row: pd.DataFrame, top_k: int = 5):
    """
    Explain a single prediction with SHAP.

    Returns:
      top_risk_factors: list of {feature, value, impact} with positive SHAP
      top_protective_factors: list of {feature, value, impact} with negative SHAP

    Notes:
      - X_row must be a DataFrame with exactly 1 row.
      - Columns must match training feature_names order.
      - SHAP 'impact' is contribution to the model output for this single example.
    """
    if not isinstance(X_row, pd.DataFrame) or len(X_row) != 1:
        raise ValueError("X_row must be a pandas DataFrame with exactly 1 row")

    explainer = _get_tree_explainer(model)

    # shap_values: (1, n_features) for binary classification
    shap_values = explainer.shap_values(X_row)
    vals = shap_values[0]
    feats = list(X_row.columns)
    values = X_row.iloc[0].values

    df = (
        pd.DataFrame({"feature": feats, "value": values, "shap": vals})
        .sort_values("shap", ascending=False)
        .reset_index(drop=True)
    )

    top_pos = df[df["shap"] > 0].head(top_k)
    top_neg = df[df["shap"] < 0].tail(top_k).sort_values("shap").reset_index(drop=True)

    def _safe_float(x):
        try:
            if pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    top_risk = [
        {
            "feature": row["feature"],
            "value": _safe_float(row["value"]),
            "impact": float(round(row["shap"], 4)),
        }
        for _, row in top_pos.iterrows()
    ]

    top_protect = [
        {
            "feature": row["feature"],
            "value": _safe_float(row["value"]),
            "impact": float(round(row["shap"], 4)),
        }
        for _, row in top_neg.iterrows()
    ]

    return top_risk, top_protect
