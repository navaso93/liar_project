"""
What is this file?
This file loads the LIAR dataset from local TSV files.

What is its responsibility?
It reads the train, valid, and test files, assigns standard LIAR column names, and returns the data as pandas DataFrames.
"""

from pathlib import Path

import pandas as pd


COLUMNS = [
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


def load_liar_split(split: str) -> pd.DataFrame:
    """
    Load one LIAR dataset split.

    Parameters
    ----------
    split : str
        One of: "train", "valid", or "test".

    Returns
    -------
    pd.DataFrame
        A DataFrame with standardized LIAR column names.
    """

    allowed_splits = ["train", "valid", "test"]

    if split not in allowed_splits:
        raise ValueError(f"split must be one of: {allowed_splits}")

    path = Path("raw_data") / f"{split}.tsv"

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
    )

    return df


def load_all_data() -> pd.DataFrame:
    """
    Load train, validation, and test splits and combine them.

    Returns
    -------
    pd.DataFrame
        Combined LIAR dataset.
    """

    df_train = load_liar_split("train")
    df_valid = load_liar_split("valid")
    df_test = load_liar_split("test")

    df = pd.concat(
        [df_train, df_valid, df_test],
        ignore_index=True,
    )

    return df
