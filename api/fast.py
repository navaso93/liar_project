"""
What is this file?
This file defines the FastAPI backend for the LIAR prediction service.

What is its responsibility?
It exposes API endpoints, receives user input, sends the input to the prediction layer, and returns prediction results as JSON.
"""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from liar.predict import SUPPORTED_MODELS, predict


app = FastAPI()


class PredictionRequest(BaseModel):
    model_name: str = "naive"
    statement: str

    subject: Optional[str] = "unknown"
    speaker: Optional[str] = "unknown"
    job_title: Optional[str] = "other"
    state: Optional[str] = "unknown"
    party: Optional[str] = "unknown"
    context: Optional[str] = "other"

    barely_true_counts: Optional[int] = 0
    false_counts: Optional[int] = 0
    half_true_counts: Optional[int] = 0
    mostly_true_counts: Optional[int] = 0
    pants_on_fire_counts: Optional[int] = 0


@app.get("/")
def root() -> dict:
    """
    Root endpoint used to check if the API is running.
    """

    return {
        "message": "LIAR prediction API is running",
        "available_models": SUPPORTED_MODELS,
    }


@app.post("/predict")
def predict_statement(request: PredictionRequest) -> dict:
    """
    Predict the truthfulness category of a user-provided statement.
    """

    result = predict(
        model_name=request.model_name,
        statement=request.statement,
        subject=request.subject,
        speaker=request.speaker,
        job_title=request.job_title,
        state=request.state,
        party=request.party,
        context=request.context,
        barely_true_counts=request.barely_true_counts,
        false_counts=request.false_counts,
        half_true_counts=request.half_true_counts,
        mostly_true_counts=request.mostly_true_counts,
        pants_on_fire_counts=request.pants_on_fire_counts,
    )

    return result
