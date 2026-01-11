import json
from sklearn.datasets import fetch_openml

ds = fetch_openml(data_id=45554, as_frame=True)
df = ds.frame

sample = df.drop(columns=["RiskPerformance"]).iloc[0].to_dict()
print(json.dumps({"data": sample}, indent=2))
