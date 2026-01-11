import pandas as pd
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


def load_df() -> pd.DataFrame:
    ds = fetch_openml(data_id=45554, as_frame=True)
    return ds.frame


def make_xy(df: pd.DataFrame):
    # target: 1 = Bad (high risk), 0 = Good
    y = (df["RiskPerformance"] == "Bad").astype(int)

    X = df.drop(columns=["RiskPerformance"])
   
    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y


def main():
    df = load_df()
    X, y = make_xy(df)

    
    X = X.fillna(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    print("\n=== XGBoost Results ===")
    print("ROC AUC:", auc)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))


    artifacts = {
        "model": model,
        "feature_names": list(X.columns)
    }
    joblib.dump(artifacts, "xgb_artifacts.joblib")
    print("\nSaved: xgb_artifacts.joblib")


if __name__ == "__main__":
    main()
