"""
What is this file?
This file contains the Naive Bayes model logic from the original LIAR notebook.

What is its responsibility?
It builds the same preprocessing-and-classification pipeline used in the notebook, trains the Naive Bayes model, and provides prediction utilities.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "statement",
    "subject",
    "context",
    "job_title",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
]

TARGET_COLUMN = "label"

SPEAKER_HISTORY_COLUMNS = [
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
]


def build_naive_pipeline() -> Pipeline:
    """
    Build the Naive Bayes pipeline based on the original notebook.

    Notebook logic:
    - TfidfVectorizer for statement
    - OneHotEncoder for subject
    - OneHotEncoder for context
    - OneHotEncoder for job_title
    - OneHotEncoder for party
    - Passthrough for speaker history counts
    - MultinomialNB(alpha=0.1)
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "statement",
                TfidfVectorizer(
                    max_df=0.9,
                    min_df=1,
                    ngram_range=(1, 1),
                ),
                "statement",
            ),
            (
                "subject",
                OneHotEncoder(handle_unknown="ignore"),
                ["subject"],
            ),
            (
                "context",
                OneHotEncoder(handle_unknown="ignore"),
                ["context"],
            ),
            (
                "job_title",
                OneHotEncoder(handle_unknown="ignore"),
                ["job_title"],
            ),
            (
                "party",
                OneHotEncoder(handle_unknown="ignore"),
                ["party"],
            ),
            (
                "speaker_history",
                "passthrough",
                SPEAKER_HISTORY_COLUMNS,
            ),
        ]
    )

    pipeline_nb = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]
    )

    return pipeline_nb


def train_naive_model(df: pd.DataFrame) -> Pipeline:
    """
    Train the Naive Bayes model on the preprocessed LIAR dataset.
    """

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    pipeline_nb = build_naive_pipeline()
    pipeline_nb.fit(X, y)

    return pipeline_nb


def build_prediction_input(
    statement: str,
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
) -> pd.DataFrame:
    """
    Build one complete LIAR-format input row for prediction.

    Even though the Naive Bayes model does not use every original LIAR column,
    the preprocessing pipeline expects the full notebook-compatible structure.
    """

    if not statement or not statement.strip():
        raise ValueError("statement is required and cannot be empty")

    input_df = pd.DataFrame(
        {
            "id": ["manual_input.json"],
            "label": ["false"],
            "statement": [statement],
            "subject": [subject],
            "speaker": [speaker],
            "job_title": [job_title],
            "state": [state],
            "party": [party],
            "barely_true_counts": [barely_true_counts],
            "false_counts": [false_counts],
            "half_true_counts": [half_true_counts],
            "mostly_true_counts": [mostly_true_counts],
            "pants_on_fire_counts": [pants_on_fire_counts],
            "context": [context],
        }
    )

    return input_df


def predict_naive(model: Pipeline, input_df: pd.DataFrame) -> dict:
    """
    Predict the class, confidence, and class probabilities for one input row.
    """

    input_df = input_df[FEATURE_COLUMNS]

    prediction = str(model.predict(input_df)[0])

    probabilities = model.predict_proba(input_df)[0]
    class_labels = model.classes_

    confidence = float(probabilities.max())

    class_probabilities = {
        str(label): float(probability)
        for label, probability in zip(class_labels, probabilities)
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
    }
