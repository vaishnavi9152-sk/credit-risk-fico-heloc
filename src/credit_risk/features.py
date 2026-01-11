import pandas as pd
from sklearn.impute import SimpleImputer


RAW_FEATURE_COLS = [
    "ExternalRiskEstimate",
    "MSinceOldestTradeOpen",
    "MSinceMostRecentTradeOpen",
    "AverageMInFile",
    "NumSatisfactoryTrades",
    "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec",
    "PercentTradesNeverDelq",
    "MSinceMostRecentDelq",
    "MaxDelq2PublicRecLast12M",
    "MaxDelqEver",
    "NumTotalTrades",
    "NumTradesOpeninLast12M",
    "PercentInstallTrades",
    "MSinceMostRecentInqexcl7days",
    "NumInqLast6M",
    "NumInqLast6Mexcl7days",
    "NetFractionRevolvingBurden",
    "NetFractionInstallBurden",
    "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance",
    "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance",
]


def _coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def fit_imputer(df: pd.DataFrame) -> SimpleImputer:
    """
    Fit the imputer on TRAINING data only.
    """
    X = df.copy()
    if "RiskPerformance" in X.columns:
        X = X.drop(columns=["RiskPerformance"])

    X = X.reindex(columns=RAW_FEATURE_COLS)
    X = _coerce_numeric(X)

    imputer = SimpleImputer(strategy="median", add_indicator=True)
    imputer.fit(X)
    return imputer


def transform_with_imputer(df: pd.DataFrame, imputer: SimpleImputer) -> pd.DataFrame:
    """
    Transform any data (train/test/inference) using a pre-fit imputer.
    """
    X = df.copy()
    if "RiskPerformance" in X.columns:
        X = X.drop(columns=["RiskPerformance"])

    X = X.reindex(columns=RAW_FEATURE_COLS)
    X = _coerce_numeric(X)

    X_imputed = imputer.transform(X)

    feature_names = list(RAW_FEATURE_COLS)
    if getattr(imputer, "indicator_", None) is not None:
        inds = imputer.indicator_.features_
        feature_names += [f"{RAW_FEATURE_COLS[i]}__missing" for i in inds]

    return pd.DataFrame(X_imputed, columns=feature_names)
