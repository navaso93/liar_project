"""
This file contains all functions related to the vectorization and embedding of the dataset,
which is needed for retrieving all the statements similar to the predicted statement.
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from liar.preprocessing import simplify_label, simplify_context

import pickle
import time

def docs_vectorstore(df_dataset):
    """
    Function not needed during the normal user workflow/process.
    This function only takes the original dataset and creates the blueprint for the embedded, vectorized vector store later.
    It basically looks at the dataset and prepares the vector store for storing it.
    """

    # This function will create the vector store blueprint (vector_store)
    # And the document to send there (documents)

    # We instantiate an embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Create a Chroma vector store using the embedder 'embeddings' we created earlier
    vector_store = Chroma(
        collection_name="liar_dataset",  # Name we give our dataset
        embedding_function=embeddings,   # Embedding function (sort of embedding blueprint)
        persist_directory="./chroma_db"  # Directory where embedded copy lives.
    )

    # Bringing all columns to uniform formats to avoid semantic duplicates due to typos or punctuation
    for col in df_dataset.columns:
        df_dataset[col] = (
            df_dataset[col]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\d+", "", regex=True) # We replace numbers by empty values
            .str.replace(r"[^\w\s]", " ", regex=True) # We replace punctuation by blank spaces, since it might separate two words.
        )

    # Applying label simplifier
    df_dataset["label"] = df_dataset["label"].apply(simplify_label)

    # Applying column cluster for "context"
    df_dataset["context"] = df_dataset["context"].apply(simplify_context)


    # Convert each dataframe row into a Langchain document
    documents = [
        Document(
            page_content=row["statement"],  # text to embed
            metadata={
                "label": row["label"],
                "speaker": row["speaker"],
                "context": row["context"]
            }
        )
        for _, row in df_dataset.iterrows()
    ]

    return vector_store, documents


def embedder(vector_store, documents, batch_size):                         # Make sure length of batch_size is not too big
    """
    This function is only required for creating the actual embedded dataset as a chroma_sqlite3 file
    The embedded dataset will be accessed in order to retrieve the most similar statements to the predicted one.
    """

    processed_batches = set()                                              # Stores completed batch numbers

    try:
        with open("processed_batches.pkl", "rb") as f:                     # Open previous progress file
            processed_batches = pickle.load(f)                             # Load completed batches
    except:
        pass                                                               # If file doesn't exist, start from scratch

    for i in range(0, len(documents), batch_size):                         # Loop through documents by batch
        batch_num = i // batch_size                                        # Compute current batch number

        if batch_num in processed_batches:                                 # Check if batch already processed
            continue                                                       # Skip if already done

        print(f"Starting batch {batch_num}")                               # Print batch being processed

        batch = documents[i:i+batch_size]                                  # Extract current batch
        vector_store.add_documents(batch)                                  # Embed and store batch in Chroma
        processed_batches.add(batch_num)                                   # Mark batch as completed

        with open("processed_batches.pkl", "wb") as f:                     # Save progress to disk
            pickle.dump(processed_batches, f)                              # Write completed batches

        time.sleep(3)   # after each successful batch

    return f"Completed batch {batch_num}"


def statement_df_retriever(vector_store, X_predict, k_similar=5, speaker=None, context=None):
    """
    This function:
    - Identifies the vector store with the embedded dataset.
    - Retrieves statement from 'X_predict' (optionally filters retrieved statements by speaker and context)
    Then it will extract from the vector store the closest stataments to 'X_predict' available.
    It will retrieve as many as specified by k_similar.
    """

    query = str(X_predict["statement"])

    chroma_filter = {}

    if speaker is not None:
        chroma_filter["speaker"] = speaker

    if context is not None:
        chroma_filter["context"] = context

    retrieved_docs = vector_store.similarity_search(
        query,
        k=k_similar,
        filter=chroma_filter if chroma_filter else None
    )

    df_retrieved = pd.DataFrame([
        {
            "speaker": doc.metadata.get("speaker"),
            "label": doc.metadata.get("label"),
            "context": doc.metadata.get("context"),
            "statement": doc.page_content
        }
        for doc in retrieved_docs
    ])

    return df_retrieved
