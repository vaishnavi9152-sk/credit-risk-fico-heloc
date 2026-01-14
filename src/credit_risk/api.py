from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from credit_risk.predict import predict_one

app = FastAPI(title="Explainable Credit Risk API")


class Applicant(BaseModel):
    data: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(applicant: Applicant):
    return predict_one(applicant.data)
