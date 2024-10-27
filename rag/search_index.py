"""The script for embedding-based retrieval."""

# Import Python Packages
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load the environment variables from the .env file
load_dotenv()

# Define the collection name and embedding model
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Access the CHROMA_PERSIST_DIR environment variable
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")

if not CHROMA_PERSIST_DIR:
    raise EnvironmentError("CHROMA_PERSIST_DIR is not set in the environment. Please make sure to define it in the .env file.")

if not os.path.exists(CHROMA_PERSIST_DIR):
    raise FileNotFoundError(f"Persist directory '{CHROMA_PERSIST_DIR}' does not exist. Please ensure the embedding index is created before retrieval.")

def main():
    # Same model as used to create persisted embedding index
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Access persisted embeddings
    db = get_embed_db(embeddings)

    # Example query for similarity indexing
    prompt = (
        "New Jersey Transit Rail Operations"
    )

    # Display matched documents and similarity scores
    print(f"Finding document matches for '{prompt}'")
    docs_scores = db.similarity_search_with_score(prompt)
    for doc, score in docs_scores:
        print(f"\nSimilarity score (lower is better): {score}")
        print(doc.metadata)
        print(doc.page_content)


def get_embed_db(embeddings):
    if CHROMA_PERSIST_DIR:
        db = get_chroma_db(embeddings, CHROMA_PERSIST_DIR)
    else:
        # Handle missing persist directory here
        raise EnvironmentError("No vector store persist directory found.")
    return db


def get_chroma_db(embeddings, persist_dir):
    # Load the Chroma vector store from the persisted directory
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return db


if __name__ == "__main__":
    main()
