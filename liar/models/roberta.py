"""
What is this file?
This file defines the RoBERTa model prediction logic for the LIAR prediction project.

What is its responsibility?
It loads a fine-tuned RoBERTa sequence classification model from disk, tokenizes an input statement, runs inference, and returns the prediction result.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "saved_models" / "roberta"
MAX_LENGTH = 256


_tokenizer = None
_model = None


def load_roberta_model():
    global _tokenizer
    global _model

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )

    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
        _model.to("cpu")
        _model.eval()

    return _tokenizer, _model


def predict(input_data: dict) -> dict:
    statement = input_data.get("statement", "")

    if not statement.strip():
        raise ValueError("Statement cannot be empty.")

    tokenizer, model = load_roberta_model()

    inputs = tokenizer(
        statement,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = F.softmax(logits, dim=1)[0]

    prediction_id = torch.argmax(probabilities).item()
    prediction = model.config.id2label[prediction_id]

    class_probabilities = {
        model.config.id2label[index]: float(probabilities[index])
        for index in range(len(probabilities))
    }

    confidence = max(class_probabilities.values())

    return {
        "model_name": "roberta",
        "prediction": prediction,
        "confidence": float(confidence),
        "class_probabilities": class_probabilities,
    }
