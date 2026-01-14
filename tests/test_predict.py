from credit_risk.predict import predict_one



def test_predict_output_shape():
    sample = {
        "ExternalRiskEstimate": 55,
        "MSinceOldestTradeOpen": 144,
        "MSinceMostRecentTradeOpen": 4,
        "AverageMInFile": 84,
        "NumSatisfactoryTrades": 20,
        "NumTrades60Ever2DerogPubRec": 3,
        "NumTrades90Ever2DerogPubRec": 0,
        "PercentTradesNeverDelq": 83,
        "MSinceMostRecentDelq": 2,
        "MaxDelq2PublicRecLast12M": 2,
        "MaxDelqEver": 5,
        "NumTotalTrades": 28,
        "NumTradesOpeninLast12M": 1,
        "PercentInstallTrades": 43,
        "MSinceMostRecentInqexcl7days": 0,
        "NumInqLast6M": 0,
        "NumInqLast6Mexcl7days": 0,
        "NetFractionRevolvingBurden": 33,
        "NetFractionInstallBurden": 0,
        "NumRevolvingTradesWBalance": 8,
        "NumInstallTradesWBalance": 1,
        "NumBank2NatlTradesWHighUtilization": 1,
        "PercentTradesWBalance": 69,
    }

    out = predict_one(sample)

    # basic structure checks
    assert "prob_bad" in out
    assert "risk_tier" in out
    assert "decision" in out
    assert "top_risk_factors" in out
    assert "top_protective_factors" in out

    # probability sanity
    assert 0 <= out["prob_bad"] <= 1
