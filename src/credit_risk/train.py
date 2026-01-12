import pandas as pd
import joblib
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

from src.credit_risk.features import fit_imputer, transform_with_imputer


def load_data() -> pd.DataFrame:
    ds = fetch_openml(data_id=45554, as_frame=True)
    return ds.frame


def pick_thresholds_cost_sensitive(
    probs: np.ndarray,
    y_true: np.ndarray,
    fn_cost: float = 3.0,
    fp_cost: float = 1.0,
) -> dict:
    """
    Find (t_low, t_high) thresholds that minimize cost.

    Tier rule:
      - prob < t_low           -> LOW  (approve)
      - t_low <= prob < t_high -> MEDIUM (manual review)
      - prob >= t_high         -> HIGH (reject)

    For cost computation we simplify and treat:
      - REJECT (prob >= t_high) as predicting BAD (1)
      - else as predicting GOOD (0)

    Cost = fn_cost * FN + fp_cost * FP
    """
    grid = np.linspace(0.05, 0.95, 91)  
    best = None

    for t_low in grid:
        for t_high in grid:
            if t_high <= t_low:
                continue

            pred_bad = (probs >= t_high).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred_bad).ravel()
            cost = fn_cost * fn + fp_cost * fp

            if best is None or cost < best["cost"]:
                best = {
                    "t_low": float(t_low),
                    "t_high": float(t_high),
                    "cost": float(cost),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }

    return best


def main():
    
    # FN = approve a bad borrower (very expensive)
    # FP = reject a good borrower (less expensive)
    FN_COST = 2.0
    FP_COST = 1.0
    

    df = load_data()

    y = (df["RiskPerformance"] == "Bad").astype(int)

    # 60/20/20 split: train/val/test
    df_trainval, df_test, y_trainval, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=42
    )
    df_train, df_val, y_train, y_val = train_test_split(
        df_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )

    imputer = fit_imputer(df_train)

    X_train = transform_with_imputer(df_train, imputer)
    X_val = transform_with_imputer(df_val, imputer)
    X_test = transform_with_imputer(df_test, imputer)

    model = XGBClassifier(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    probs_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs_test)

    print("\n=== XGBoost (train-fit preprocessing, proper split) ===")
    print("ROC AUC (test):", auc)

    probs_val = model.predict_proba(X_val)[:, 1]
    best = pick_thresholds_cost_sensitive(
        probs=probs_val,
        y_true=y_val.to_numpy(),
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )

    print(f"\n=== Thresholds chosen on VALIDATION (FN cost={FN_COST}, FP cost={FP_COST}) ===")
    print(best)

    t_low, t_high = best["t_low"], best["t_high"]

    pred_bad_test = (probs_test >= t_high).astype(int)

    print("\n=== TEST performance at chosen t_high (reject as BAD) ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred_bad_test))
    print("\nClassification Report:\n", classification_report(y_test, pred_bad_test))

    artifacts = {
        "model": model,
        "imputer": imputer,
        "feature_names": list(X_train.columns),
        "thresholds": {"t_low": t_low, "t_high": t_high},
        "costs": {"fn_cost": FN_COST, "fp_cost": FP_COST},
    }
    joblib.dump(artifacts, "artifacts.joblib")
    print("\nSaved: artifacts.joblib (with thresholds + costs)")


if __name__ == "__main__":
    main()
