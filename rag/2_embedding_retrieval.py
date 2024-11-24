"""PART II: embedding-based retrieval."""

# Import Python packages
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
    """Main function to perform similarity search on the persisted vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL) # Use same model
    db = get_embed_db(embeddings)

    # Example query for similarity indexing
    prompt = (
        "EO 13773"
    )

    # Display matched documents and similarity scores
    print(f"Finding document matches for '{prompt}'")
    docs_scores = db.similarity_search_with_score(prompt)
    for doc, score in docs_scores:
        print(f"\nSimilarity score (lower is better): {score:.2f}")
        print(doc.metadata)
        print(doc.page_content)


def get_embed_db(embeddings):
    """Retrieves the Chroma vector store using the specified embedding model."""
    if CHROMA_PERSIST_DIR:
        db = get_chroma_db(embeddings, CHROMA_PERSIST_DIR)
    else:
        raise EnvironmentError("No vector store persist directory found.")
    return db


def get_chroma_db(embeddings, persist_dir):
    """Load the Chroma vector store from the persisted directory."""
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return db


if __name__ == "__main__":
    main()
