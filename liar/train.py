"""
What is this file?
This file is the training entry point for the LIAR project.

What is its responsibility?
It loads the dataset, applies preprocessing, trains the selected model, and saves the trained model to disk.
"""

from liar.data import load_all_data
from liar.preprocessing import preprocess_data
from liar.models.naive import train_naive_model
from liar.registry import save_model


SUPPORTED_MODELS = ["naive"]


def train(model_name: str = "naive") -> None:
    """
    Train and save a supported model.

    Parameters
    ----------
    model_name : str
        Name of the model to train. Currently supported: "naive".
    """

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported models: {SUPPORTED_MODELS}")

    print(f"🚀 Training started for model: {model_name}")

    print("📥 Loading data...")
    df = load_all_data()
    print(f"✅ Data loaded: {df.shape}")

    print("🧹 Preprocessing data...")
    df = preprocess_data(df)
    print(f"✅ Data preprocessed: {df.shape}")

    if model_name == "naive":
        print("🤖 Training Naive Bayes model...")
        model = train_naive_model(df)

    save_model(model, model_name)

    print(f"✅ Training completed for model: {model_name}")


if __name__ == "__main__":
    train("naive")
