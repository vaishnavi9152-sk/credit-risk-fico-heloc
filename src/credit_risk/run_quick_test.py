from sklearn.datasets import fetch_openml
from credit_risk.predict import predict_one

ds = fetch_openml(data_id=45554, as_frame=True)
df = ds.frame

sample = df.drop(columns=["RiskPerformance"]).iloc[0].to_dict()
print(predict_one(sample))
