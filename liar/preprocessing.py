"""
What is this file?
This file contains the full preprocessing logic from the original LIAR notebook.

What is its responsibility?
It applies the same cleaning, category simplification, label grouping, state correction, tokenization, stopword removal, lemmatization, and text reconstruction used in the notebook.
"""

import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from rapidfuzz import process


TEXT_COLUMNS = [
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state",
    "party",
    "context",
]


TOKENIZE_COLUMNS = [
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state",
    "context",
]


SPEAKER_HISTORY_COLUMNS = [
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
]


VALID_STATES = [
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
]


def ensure_nltk_resources() -> None:
    """
    Ensure that the NLTK resources required by the notebook preprocessing are available.
    """

    required_resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, package_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name)


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the notebook text cleaning logic:
    - fill missing values
    - lowercase
    - strip spaces
    - remove numbers
    - replace punctuation and symbols with spaces
    """

    df = df.copy()

    for col in TEXT_COLUMNS:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\d+", "", regex=True)
            .str.replace(r"[^\w\s]", " ", regex=True)
        )

    df["id"] = df["id"].astype(str).str.removesuffix(".json")

    return df


def correct_state(state: str) -> str:
    """
    Correct state names by matching them to the closest valid US state name.
    """

    if pd.isna(state):
        return state

    state = str(state).lower().strip()

    if state == "":
        return state

    match, score, _ = process.extractOne(state, VALID_STATES)

    if score >= 80:
        return match

    return state


def simplify_job_title(title: str) -> str:
    """
    Simplify job titles into broader categories.
    """

    if pd.isna(title):
        return "unknown"

    title = str(title).lower().strip()

    if title in ["president", "president elect", "presidential candidate", "former president"]:
        return "president"

    elif "governor" in title:
        return "governor"

    elif "senator" in title:
        return "senator"

    elif "legislator" in title:
        return "legislator"

    elif "mayor" in title:
        return "mayor"

    elif "ambassador" in title:
        return "ambassador"

    elif any(x in title for x in ["representative", "congressman", "congresswoman", "house"]):
        return "representative"

    elif "attorney" in title or "lawyer" in title:
        return "legal"

    elif any(x in title for x in ["radio host", "host", "journalist", "reporter", "blog", "social", "columnist"]):
        return "media"

    elif any(x in title for x in ["president", "ceo", "director", "founder", "business"]):
        return "private executive"

    elif "candidate" in title:
        return "candidate"

    else:
        return "other"


def simplify_context(context: str) -> str:
    """
    Simplify context values into broader categories.
    """

    if pd.isna(context):
        return "unknown"

    context = str(context).lower()

    if any(x in context for x in ["ad", "commercial", "mailer"]):
        return "ad"

    elif "interview" in context:
        return "interview"

    elif any(x in context for x in ["press release", "news release"]):
        return "press release"

    elif any(x in context for x in ["press conference", "news conference"]):
        return "news conference"

    elif "debate" in context:
        return "debate"

    elif any(x in context for x in ["tweet", "facebook", "website", "social"]):
        return "social media"

    elif "statement" in context:
        return "statement"

    elif any(x in context for x in ["email", " e mail", "chain mail"]):
        return "email"

    elif any(x in context for x in ["television", "tv", "fox news", "cnn", "meet the press"]):
        return "tv appearance"

    else:
        return "other"


def simplify_party(party: str) -> str:
    """
    Simplify party affiliations into broader categories.
    """

    if pd.isna(party):
        return "unknown"

    party = str(party).lower()

    if any(x in party for x in ["republican", "tea party"]):
        return "republican"

    elif any(x in party for x in ["democrat", "democratic farmer labor"]):
        return "democrat"

    elif any(x in party for x in ["independent", "libertarian", "green", "constitution party", "moderate"]):
        return "other_party"

    elif any(
        x in party
        for x in [
            "organization",
            "journalist",
            "columnist",
            "newsmaker",
            "activist",
            "talk show host",
            "state official",
            "education official",
            "government body",
            "business leader",
            "labor leader",
            "commissioner",
        ]
    ):
        return "organization"

    elif party == "none":
        return "unknown"

    else:
        return "other"


def simplify_subject(subject: str) -> str:
    """
    Simplify subjects into broader topic groups.
    """

    if pd.isna(subject):
        return "unknown"

    subject = str(subject).lower()

    if any(x in subject for x in ["health", "medicare", "medicaid", "hospital", "disease", "ebola"]):
        return "health"

    elif any(x in subject for x in ["economy", "job", "worker", "small business", "income"]):
        return "economy"

    elif "tax" in subject:
        return "taxation"

    elif any(x in subject for x in ["budget", "finance", "deficit", "debt"]):
        return "budget"

    elif "education" in subject:
        return "education"

    elif any(x in subject for x in ["immigration", "border", "refugee"]):
        return "immigration"

    elif any(x in subject for x in ["election", "candidate biography", "campaign", "voting record"]):
        return "elections"

    elif any(x in subject for x in ["crime", "criminal justice", "terrorism", "gun", "public safety"]):
        return "security"

    elif any(x in subject for x in ["foreign policy", "iraq", "afghanistan", "china", "israel", "trade"]):
        return "foreign_affairs"

    elif any(x in subject for x in ["environment", "climate", "weather"]):
        return "environment"

    elif "energy" in subject:
        return "energy"

    elif "transportation" in subject:
        return "transportation"

    elif any(x in subject for x in ["military", "veteran", "defense"]):
        return "military"

    elif any(x in subject for x in ["government", "ethic", "legal issue", "congress"]):
        return "government"

    elif any(x in subject for x in ["abortion", "religion", "gay", "lesbian", "woman", "queer", "marriage"]):
        return "social_issues"

    elif any(x in subject for x in ["science", "technology", "research"]):
        return "science"

    else:
        return "other"


def simplify_label(label: str) -> str:
    """
    Convert the original 6 LIAR labels into the 3 classes used in the notebook.
    """

    if label in ["true", "mostly-true"]:
        return "trustworthy"

    elif label == "half-true":
        return "questionable"

    else:
        return "unreliable"


def fill_speaker_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing speaker history count values with zero.
    """

    df = df.copy()
    df[SPEAKER_HISTORY_COLUMNS] = df[SPEAKER_HISTORY_COLUMNS].fillna(0)

    return df


def tokenize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tokenize the same text columns used in the notebook.
    """

    df = df.copy()

    for col in TOKENIZE_COLUMNS:
        df[col] = df[col].fillna("").apply(word_tokenize)

    return df


def remove_stopwords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove English stopwords from tokenized text columns.
    """

    df = df.copy()
    stop_words = set(stopwords.words("english"))

    for col in TOKENIZE_COLUMNS:
        df[col] = df[col].apply(lambda tokens: [word for word in tokens if word not in stop_words])

    return df


def lemmatize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lemmatize tokenized text columns using WordNetLemmatizer.
    """

    df = df.copy()
    lemmatizer = WordNetLemmatizer()

    for col in TOKENIZE_COLUMNS:
        df[col] = df[col].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

    return df


def join_tokenized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert token lists back into strings, matching the final notebook preprocessing step.
    """

    df = df.copy()

    for col in TOKENIZE_COLUMNS:
        df[col] = df[col].apply(lambda tokens: " ".join(tokens))

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline from the original notebook.
    """

    ensure_nltk_resources()

    df = df.copy()

    df = clean_text_columns(df)

    df["state"] = df["state"].apply(correct_state)
    df["job_title"] = df["job_title"].apply(simplify_job_title)
    df["context"] = df["context"].apply(simplify_context)
    df["party"] = df["party"].apply(simplify_party)
    df["subject"] = df["subject"].apply(simplify_subject)
    df["label"] = df["label"].apply(simplify_label)

    df = fill_speaker_history(df)

    df = tokenize_text_columns(df)
    df = remove_stopwords(df)
    df = lemmatize_text_columns(df)
    df = join_tokenized_columns(df)

    return df
