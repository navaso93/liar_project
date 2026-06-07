"""
What is this file?
This file is the prediction entry point for the LIAR project.

What is its responsibility?
It routes prediction requests to the correct trained model, prepares the required input format, and returns a consistent prediction response.
"""

from liar.preprocessing import preprocess_data
from liar.registry import load_model
from liar.models.naive import build_prediction_input, predict_naive
from liar.models.naive_xboost import predict as predict_naive_xboost


SUPPORTED_MODELS = ["naive", "naive_xboost"]


def predict(
    statement: str,
    model_name: str = "naive",
    subject: str = "unknown",
    speaker: str = "unknown",
    job_title: str = "other",
    state: str = "unknown",
    party: str = "unknown",
    context: str = "other",
    barely_true_counts: int = 0,
    false_counts: int = 0,
    half_true_counts: int = 0,
    mostly_true_counts: int = 0,
    pants_on_fire_counts: int = 0,
) -> dict:
    """
    Predict the truthfulness category of one statement.
    """

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_name: {model_name}. Supported models: {SUPPORTED_MODELS}"
        )

    if model_name == "naive":
        input_df = build_prediction_input(
            statement=statement,
            subject=subject,
            speaker=speaker,
            job_title=job_title,
            state=state,
            party=party,
            context=context,
            barely_true_counts=barely_true_counts,
            false_counts=false_counts,
            half_true_counts=half_true_counts,
            mostly_true_counts=mostly_true_counts,
            pants_on_fire_counts=pants_on_fire_counts,
        )

        input_df = preprocess_data(input_df)

        model = load_model(model_name)

        result = predict_naive(model, input_df)

        return {
            "model_name": model_name,
            **result,
        }

    if model_name == "naive_xboost":
        result = predict_naive_xboost(
            {
                "statement": statement,
                "speaker": speaker,
                "context": context,
            }
        )

        return result

    raise ValueError(f"Unsupported model_name: {model_name}")


if __name__ == "__main__":
    naive_result = predict(
        statement="The economy is growing faster than ever.",
        model_name="naive",
    )

    print("Naive result:")
    print(naive_result)

    naive_xboost_result = predict(
        statement="Donald Trump was born on Mars.",
        speaker="Barack Obama",
        context="interview",
        model_name="naive_xboost",
    )

    print("Naive-XGBoost result:")
    print(naive_xboost_result)
