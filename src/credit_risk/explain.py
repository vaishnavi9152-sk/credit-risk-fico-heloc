import numpy as np
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


def _coerce_numeric_df(X_row: pd.DataFrame) -> pd.DataFrame:
    """
    Force everything to float for SHAP to avoid CI dtype issues.
    Any non-parsable values become NaN -> filled with 0.0.
    """
    X_num = X_row.copy()

    # Convert every column safely to numeric
    for c in X_num.columns:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")

    # Fill remaining NaNs and cast to float
    X_num = X_num.fillna(0.0).astype(np.float64)
    return X_num


def explain_one(model, X_row: pd.DataFrame, top_k: int = 5):
    """
    Explain a single prediction with SHAP.

    Returns:
      top_risk_factors: list of {feature, value, impact} with positive SHAP
      top_protective_factors: list of {feature, value, impact} with negative SHAP

    Notes:
      - X_row must be a DataFrame with exactly 1 row.
      - Columns must match training feature_names order.
    """
    if not isinstance(X_row, pd.DataFrame) or len(X_row) != 1:
        raise ValueError("X_row must be a pandas DataFrame with exactly 1 row")

    # Keep original values for reporting
    feats = list(X_row.columns)
    raw_values = X_row.iloc[0].to_dict()

    # Force numeric for SHAP (prevents CI conversion errors)
    X_num = _coerce_numeric_df(X_row)

    explainer = _get_tree_explainer(model)

    shap_values = explainer.shap_values(X_num)
    sv = np.array(shap_values)

    # Handle possible shapes:
    # (1, n_features) OR (2, 1, n_features)
    if sv.ndim == 3:
        sv = sv[1]  # class 1
    vals = sv[0]

    df = (
        pd.DataFrame({"feature": feats, "shap": vals})
        .sort_values("shap", ascending=False)
        .reset_index(drop=True)
    )

    top_pos = df[df["shap"] > 0].head(top_k)
    top_neg = df[df["shap"] < 0].tail(top_k).sort_values("shap").reset_index(drop=True)

    def _safe_float(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            return float(x)
        except Exception:
            return None

    top_risk = [
        {
            "feature": row["feature"],
            "value": _safe_float(raw_values.get(row["feature"])),
            "impact": float(round(row["shap"], 4)),
        }
        for _, row in top_pos.iterrows()
    ]

    top_protect = [
        {
            "feature": row["feature"],
            "value": _safe_float(raw_values.get(row["feature"])),
            "impact": float(round(row["shap"], 4)),
        }
        for _, row in top_neg.iterrows()
    ]

    return top_risk, top_protect
