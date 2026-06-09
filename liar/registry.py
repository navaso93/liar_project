"""
What is this file?
This file manages saving and loading trained models for the LIAR project.

What is its responsibility?
It stores trained model objects on disk and loads them back when the API or prediction layer needs them.
"""

from pathlib import Path

import joblib


SAVED_MODELS_DIR = Path("saved_models")

MODEL_PATHS = {
    "naive": SAVED_MODELS_DIR / "naive" / "model.joblib",
    "naive_xboost": SAVED_MODELS_DIR / "naive_xboost" / "model.joblib",
}


def save_model(model, model_name: str) -> Path:
    """
    Save a trained model to disk.
    """

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model_path = MODEL_PATHS[model_name]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    print(f"✅ Model saved to: {model_path}")

    return model_path


def load_model(model_name: str):
    """
    Load a trained model from disk.
    """

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model_path = MODEL_PATHS[model_name]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    print(f"✅ Model loaded from: {model_path}")

    return model
