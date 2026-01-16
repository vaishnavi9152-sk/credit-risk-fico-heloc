import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

dataset = fetch_openml(data_id=45554, as_frame=True)
df = dataset.frame

df["target"] = (df["RiskPerformance"] == "Bad").astype(int)

X = df.drop(columns=["RiskPerformance", "target"])
y = df["target"]

X = X.apply(pd.to_numeric, errors="coerce")

X = X.fillna(-1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, probs))
print(classification_report(y_test, model.predict(X_test)))
