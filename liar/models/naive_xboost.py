"""
What is this file?
This file defines the Naive-XGBoost model for the LIAR prediction project.

What is its responsibility?
It trains a Naive Bayes model first, uses its unreliable-class probability as an extra feature, then trains an XGBoost classifier and saves both models together.
"""

from pathlib import Path

import joblib
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from rapidfuzz import process

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
MODEL_DIR = PROJECT_ROOT / "saved_models" / "naive_xboost"
MODEL_PATH = MODEL_DIR / "model.joblib"


LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]


def load_liar_data() -> pd.DataFrame:
    train_df = pd.read_csv(
        RAW_DATA_DIR / "train.tsv",
        sep="\t",
        header=None,
        names=LIAR_COLUMNS,
    )

    test_df = pd.read_csv(
        RAW_DATA_DIR / "test.tsv",
        sep="\t",
        header=None,
        names=LIAR_COLUMNS,
    )

    valid_df = pd.read_csv(
        RAW_DATA_DIR / "valid.tsv",
        sep="\t",
        header=None,
        names=LIAR_COLUMNS,
    )

    return pd.concat([train_df, test_df, valid_df], ignore_index=True)


def load_politifact_data() -> pd.DataFrame:
    politifact_df = pd.read_json(
        RAW_DATA_DIR / "politifact_factcheck_data.json",
        lines=True,
    )

    politifact_df["verdict"] = politifact_df["verdict"].replace(
        {"mostly-false": "barely-true"}
    )

    politifact_df = politifact_df.rename(
        columns={
            "verdict": "label",
            "statement_originator": "speaker",
            "statement_source": "context",
        }
    )

    return politifact_df[["label", "speaker", "statement", "context"]]


def load_training_data() -> pd.DataFrame:
    liar_df = load_liar_data()
    politifact_df = load_politifact_data()

    dataset = pd.concat([liar_df, politifact_df], ignore_index=True)

    return dataset[["label", "speaker", "statement", "context"]]


def simplify_label(label: str) -> str:
    if label in ["trustworthy", "questionable", "unreliable"]:
        return label

    if label in ["true", "mostly-true"]:
        return "trustworthy"

    if label == "half-true":
        return "questionable"

    return "unreliable"


def simplify_context(context: str) -> str:
    if pd.isna(context):
        return "unknown"

    context = str(context).lower()

    if any(x in context for x in ["ad", "commercial", "mailer"]):
        return "ad"

    if "interview" in context:
        return "interview"

    if any(x in context for x in ["press release", "news release"]):
        return "press release"

    if any(x in context for x in ["press conference", "news conference"]):
        return "news conference"

    if "debate" in context:
        return "debate"

    if any(x in context for x in ["tweet", "facebook", "website", "social"]):
        return "social media"

    if "statement" in context:
        return "statement"

    if any(x in context for x in ["email", " e mail", "chain mail"]):
        return "email"

    if any(x in context for x in ["television", "tv", "fox news", "cnn", "meet the press"]):
        return "tv appearance"

    return "other"


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    text_columns = ["statement", "speaker", "context"]

    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.strip()
                .str.replace(r"\d+", "", regex=True)
                .str.replace(r"[^\w\s]", " ", regex=True)
            )

    df["statement"] = df["speaker"] + " " + df["statement"]

    df = df.drop(columns=["speaker"])

    return df


def tokenize_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    token_columns = ["statement", "context"]

    for column in token_columns:
        if column in df.columns:
            df[column] = df[column].fillna("").apply(word_tokenize)

    return df


def remove_stopwords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    stop_words = set(stopwords.words("english"))
    token_columns = ["statement", "context"]

    for column in token_columns:
        if column in df.columns:
            df[column] = df[column].apply(
                lambda tokens: [word for word in tokens if word not in stop_words]
            )

    return df


def lemmatize_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lemmatizer = WordNetLemmatizer()
    token_columns = ["statement", "context"]

    for column in token_columns:
        if column in df.columns:
            df[column] = df[column].apply(
                lambda tokens: [lemmatizer.lemmatize(word) for word in tokens]
            )

    return df


def stringify_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    token_columns = ["statement", "context"]

    for column in token_columns:
        if column in df.columns:
            df[column] = df[column].apply(lambda tokens: " ".join(tokens))

    return df


def preprocess_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["label"] = df["label"].apply(simplify_label)

    df = clean_text_columns(df)

    if "context" in df.columns:
        df["context"] = df["context"].apply(simplify_context)

    df = tokenize_text(df)
    df = remove_stopwords(df)
    df = lemmatize_text(df)
    df = stringify_tokens(df)

    return df

def split_data(df: pd.DataFrame):
    y = df["label"]
    X = df.drop(columns=["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def train_naive_bayes(X_train: pd.DataFrame, y_train: pd.Series):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "statement",
                TfidfVectorizer(
                    max_df=0.9,
                    min_df=1,
                    ngram_range=(1, 3),
                ),
                "statement",
            ),
            (
                "context",
                OneHotEncoder(handle_unknown="ignore"),
                ["context"],
            ),
        ]
    )

    pipeline_nb = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]
    )

    probas = cross_val_predict(
        pipeline_nb,
        X_train,
        y_train,
        cv=5,
        method="predict_proba",
        n_jobs=-1,
    )

    pipeline_nb.fit(X_train, y_train)

    unreliable_class_index = list(pipeline_nb.classes_).index("unreliable")

    unrel_proba = pd.DataFrame(
        probas[:, unreliable_class_index],
        columns=["unrel_proba"],
        index=X_train.index,
    )

    return pipeline_nb, unrel_proba


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    unrel_proba: pd.DataFrame,
):
    X_train_without_statement = X_train.drop(columns=["statement"])

    X_extended = pd.concat(
        [X_train_without_statement, unrel_proba],
        axis=1,
    )

    preprocessor_xgb = ColumnTransformer(
        transformers=[
            (
                "context",
                OneHotEncoder(handle_unknown="ignore"),
                ["context"],
            ),
            (
                "unrel_proba",
                "passthrough",
                ["unrel_proba"],
            ),
        ]
    )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    pipeline_xgb = Pipeline(
        steps=[
            ("preprocessor", preprocessor_xgb),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.3,
                    subsample=0.1,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="mlogloss",
                ),
            ),
        ]
    )

    pipeline_xgb.fit(X_extended, y_train_encoded)

    return pipeline_xgb, label_encoder

def evaluate_naive_xboost(
    pipeline_nb: Pipeline,
    pipeline_xgb: Pipeline,
    label_encoder: LabelEncoder,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    nb_probas = pipeline_nb.predict_proba(X_test)

    unreliable_class_index = list(pipeline_nb.classes_).index("unreliable")

    unrel_proba_test = pd.DataFrame(
        nb_probas[:, unreliable_class_index],
        columns=["unrel_proba"],
        index=X_test.index,
    )

    X_test_without_statement = X_test.drop(columns=["statement"])

    X_extended = pd.concat(
        [X_test_without_statement, unrel_proba_test],
        axis=1,
    )

    y_test_encoded = label_encoder.transform(y_test)

    y_pred_encoded = pipeline_xgb.predict(X_extended)

    return {
        "accuracy": accuracy_score(y_test_encoded, y_pred_encoded),
        "recall_macro": recall_score(
            y_test_encoded,
            y_pred_encoded,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_test_encoded,
            y_pred_encoded,
            average="macro",
            zero_division=0,
        ),
    }

def save_model(
    pipeline_nb: Pipeline,
    pipeline_xgb: Pipeline,
    label_encoder: LabelEncoder,
    metrics: dict,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "pipeline_nb": pipeline_nb,
        "pipeline_xgb": pipeline_xgb,
        "label_encoder": label_encoder,
        "metrics": metrics,
    }

    joblib.dump(model_bundle, MODEL_PATH)


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model first."
        )

    return joblib.load(MODEL_PATH)

def train_and_save_model() -> dict:
    df = load_training_data()
    df = preprocess_training_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    pipeline_nb, unrel_proba_train = train_naive_bayes(X_train, y_train)

    pipeline_xgb, label_encoder = train_xgboost(
        X_train,
        y_train,
        unrel_proba_train,
    )

    metrics = evaluate_naive_xboost(
        pipeline_nb,
        pipeline_xgb,
        label_encoder,
        X_test,
        y_test,
    )

    save_model(
        pipeline_nb,
        pipeline_xgb,
        label_encoder,
        metrics,
    )

    return metrics

def preprocess_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_columns = {
        "statement": "unknown",
        "speaker": "unknown",
        "context": "unknown",
    }

    for column, default_value in required_columns.items():
        if column not in df.columns:
            df[column] = default_value

    df = df[["speaker", "statement", "context"]]

    df = clean_text_columns(df)

    if "context" in df.columns:
        df["context"] = df["context"].apply(simplify_context)

    df = tokenize_text(df)
    df = remove_stopwords(df)
    df = lemmatize_text(df)
    df = stringify_tokens(df)

    return df


def predict(input_data: dict) -> dict:
    model_bundle = load_model()

    pipeline_nb = model_bundle["pipeline_nb"]
    pipeline_xgb = model_bundle["pipeline_xgb"]
    label_encoder = model_bundle["label_encoder"]

    X_predict = pd.DataFrame([input_data])

    X_predict = preprocess_prediction_data(X_predict)

    nb_probas = pipeline_nb.predict_proba(X_predict)

    unreliable_class_index = list(pipeline_nb.classes_).index("unreliable")

    unrel_proba = pd.DataFrame(
        nb_probas[:, unreliable_class_index],
        columns=["unrel_proba"],
        index=X_predict.index,
    )

    X_predict_without_statement = X_predict.drop(columns=["statement"])

    X_extended = pd.concat(
        [X_predict_without_statement, unrel_proba],
        axis=1,
    )

    prediction_encoded = pipeline_xgb.predict(X_extended)
    probabilities = pipeline_xgb.predict_proba(X_extended)[0]

    prediction = label_encoder.inverse_transform(prediction_encoded)[0]

    class_probabilities = {
        label: float(probability)
        for label, probability in zip(label_encoder.classes_, probabilities)
    }

    confidence = max(class_probabilities.values())

    return {
        "model_name": "naive_xboost",
        "prediction": prediction,
        "confidence": float(confidence),
        "class_probabilities": class_probabilities,
    }
