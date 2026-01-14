from fastapi.testclient import TestClient
from credit_risk.api import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict_endpoint():
    payload = {
        "data": {
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
    }

    res = client.post("/predict", json=payload)
    assert res.status_code == 200, res.text

    data = res.json()
    assert "prob_bad" in data
    assert "decision" in data
    assert "top_risk_factors" in data
    assert "top_protective_factors" in data
