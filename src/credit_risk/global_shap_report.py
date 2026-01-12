from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

import shap
from sklearn.datasets import fetch_openml

from src.credit_risk.features import transform_with_imputer


def load_data() -> pd.DataFrame:
    ds = fetch_openml(data_id=45554, as_frame=True)
    return ds.frame


def main(
    artifacts_path: str = "artifacts.joblib",
    out_dir: str = "reports",
    sample_size: int = 2000,
    top_k_plot: int = 15,
    random_state: int = 42,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    
    art = joblib.load(artifacts_path)
    model = art["model"]
    imputer = art["imputer"]
    feature_names = art["feature_names"]

    
    df = load_data()

    if "RiskPerformance" in df.columns:
        df = df.drop(columns=["RiskPerformance"])

    
    X = transform_with_imputer(df, imputer)

    
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]

    
    if sample_size is not None and sample_size < len(X):
        X_sample = X.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X

    # Global SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    
    sv = np.array(shap_values)
    if sv.ndim == 3:
        sv = sv[1]

    mean_abs = np.abs(sv).mean(axis=0)

    global_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    csv_path = out / "global_shap.csv"
    global_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot top-K
    top = global_df.head(top_k_plot).iloc[::-1]  
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value| (global importance)")
    plt.title(f"Global Feature Importance (Top {top_k_plot})")
    plt.tight_layout()

    png_path = out / f"global_shap_top{top_k_plot}.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
