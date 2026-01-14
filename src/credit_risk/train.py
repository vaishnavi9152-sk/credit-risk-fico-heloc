import pandas as pd
import joblib
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
)

from xgboost import XGBClassifier

from credit_risk.features import fit_imputer, transform_with_imputer


def load_data() -> pd.DataFrame:
    ds = fetch_openml(data_id=45554, as_frame=True)
    return ds.frame


def pick_thresholds_cost_sensitive(
    probs: np.ndarray,
    y_true: np.ndarray,
    fn_cost: float = 2.0,
    fp_cost: float = 1.0,
) -> dict:
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
    FN_COST = 2.0
    FP_COST = 1.0

    df = load_data()
    y = (df["RiskPerformance"] == "Bad").astype(int)

    # 80/20: trainval/test
    df_trainval, df_test, y_trainval, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=42
    )

    # Split trainval into TRAIN (60%) and THRESH (20%)
    df_train, df_thresh, y_train, y_thresh = train_test_split(
        df_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )

    # preprocessing fit only on TRAIN
    imputer = fit_imputer(df_train)
    X_train = transform_with_imputer(df_train, imputer)
    X_thresh = transform_with_imputer(df_thresh, imputer)
    X_test = transform_with_imputer(df_test, imputer)

    # model
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

    # evaluate probability quality (uncalibrated)
    probs_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs_test)
    brier = brier_score_loss(y_test, probs_test)

    print("\n=== XGBoost (uncalibrated) ===")
    print("ROC AUC (test):", auc)
    print("Brier (test):", brier)

    # choose thresholds on THRESH
    probs_thresh = model.predict_proba(X_thresh)[:, 1]
    best = pick_thresholds_cost_sensitive(
        probs=probs_thresh,
        y_true=y_thresh.to_numpy(),
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )

    print(f"\n=== Thresholds chosen on THRESH (FN cost={FN_COST}, FP cost={FP_COST}) ===")
    print(best)
    t_low, t_high = best["t_low"], best["t_high"]

    # final test behavior using t_high (reject as BAD)
    pred_bad_test = (probs_test >= t_high).astype(int)

    print("\n=== TEST performance at chosen t_high (reject as BAD) ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred_bad_test))
    print("\nClassification Report:\n", classification_report(y_test, pred_bad_test))

    artifacts = {
        "model": model,  # base model used in API
        "imputer": imputer,
        "feature_names": list(X_train.columns),
        "thresholds": {"t_low": t_low, "t_high": t_high},
        "costs": {"fn_cost": FN_COST, "fp_cost": FP_COST},
        "metrics": {"auc_test": float(auc), "brier_test": float(brier)},
        "calibration": {"used": False, "note": "sigmoid/isotonic tested; Brier worsened"},
    }
    joblib.dump(artifacts, "artifacts.joblib")
    print("\nSaved: artifacts.joblib (base model + thresholds + costs + metrics)")


if __name__ == "__main__":
    main()
