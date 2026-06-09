"""
What is this file?
This file manages the Chroma vector store used to retrieve statements similar to a user-provided statement.

What is its responsibility?
It loads an existing Chroma database from disk, embeds the input statement with Gemini embeddings, retrieves similar statements, and returns them in a JSON-friendly format.
"""

from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "liar_dataset"
EMBEDDING_MODEL = "models/gemini-embedding-001"


_vector_store = None


def load_vectorstore() -> Chroma:
    """
    Load the existing Chroma vector store from disk.
    """

    global _vector_store

    if _vector_store is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Chroma database folder not found at: {CHROMA_DIR}"
            )

        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
        )

        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    return _vector_store


def retrieve_similar_statements(
    statement: str,
    k_similar: int = 4,
    speaker: str | None = None,
    context: str | None = None,
) -> pd.DataFrame:
    """
    Retrieve the most similar statements from the Chroma vector store.
    """

    if not statement.strip():
        raise ValueError("Statement cannot be empty.")

    vector_store = load_vectorstore()

    chroma_filter = {}

    if speaker:
        chroma_filter["speaker"] = speaker.lower().strip()

    if context:
        chroma_filter["context"] = context.lower().strip()

    retrieved_docs = vector_store.similarity_search(
        statement,
        k=k_similar,
        filter=chroma_filter if chroma_filter else None,
    )

    retrieved_rows = [
        {
            "speaker": doc.metadata.get("speaker", "unknown"),
            "label": doc.metadata.get("label", "unknown"),
            "context": doc.metadata.get("context", "unknown"),
            "statement": doc.page_content,
        }
        for doc in retrieved_docs
    ]

    return pd.DataFrame(retrieved_rows)


def retrieve_similar_statements_as_dicts(
    statement: str,
    k_similar: int = 4,
    speaker: str | None = None,
    context: str | None = None,
) -> list[dict]:
    """
    Retrieve similar statements and return them as a list of dictionaries.
    """

    df_retrieved = retrieve_similar_statements(
        statement=statement,
        k_similar=k_similar,
        speaker=speaker,
        context=context,
    )

    return df_retrieved.to_dict(orient="records")
