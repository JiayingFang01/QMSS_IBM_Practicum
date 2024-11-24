"""PART II: embedding-based retrieval."""

# Import Python packages
import os
import re
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
    """Main function to perform hybrid search on the persisted vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)  # Use the same model
    db = get_embed_db(embeddings)

    # Example query
    prompt = """
    What are the key points of Executive Order 13773?
    """

    # Perform hybrid search
    print(f"Finding document matches for:\n{prompt}")
    results = hybrid_search_with_metadata(prompt, db)

    # Display results
    for doc, score in results:
        print(f"\nSimilarity score: {score:.2f}")
        print(doc.metadata)
        print(doc.page_content)


def extract_keywords(prompt):
    """Extract keywords for metadata fields from the query, including similar terms and inferred values."""
    keywords = {}

    # Extract Executive Order number or similar terms
    eo_match = re.search(r"(EO|Executive Order|Presidential Order|Order)\s*(\d+)", prompt, re.IGNORECASE)
    if eo_match:
        keywords["executive_order_number"] = eo_match.group(2)

    # Extract President's name
    president_match = re.search(
        r"President\s+(Joseph\s+Biden|Barack\s+Obama|Donald\s+Trump|George\s+(W\s+)?Bush|Bill\s+Clinton|Ronald\s+Reagan)",
        prompt, re.IGNORECASE
    )
    if president_match:
        keywords["president"] = president_match.group(1)

    # Extract Publication Date or similar terms
    publication_date_match = re.search(
        r"(published|released|effective|publication date|date of issuance)\s+(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2})",
        prompt, re.IGNORECASE
    )
    if publication_date_match:
        keywords["publication_date"] = publication_date_match.group(2)

    # Extract Signing Date or similar terms
    signing_date_match = re.search(
        r"(signed on|signing date|issued on)\s+(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2})",
        prompt, re.IGNORECASE
    )
    if signing_date_match:
        keywords["signing_date"] = signing_date_match.group(2)

    # Infer Year from Dates
    year_match = re.search(r"\b(19|20)\d{2}\b", prompt)
    if year_match:
        keywords["year"] = year_match.group(0)

    # Extract Document Number or similar terms
    document_number_match = re.search(r"(Document Number|Doc No)\s+(\d+)", prompt, re.IGNORECASE)
    if document_number_match:
        keywords["document_number"] = document_number_match.group(2)

    # Extract Title (General Topic or Subject)
    title_match = re.search(r"(on|about|regarding)\s+([\w\s]+)", prompt, re.IGNORECASE)
    if title_match:
        keywords["title"] = title_match.group(2).strip()

    return keywords


def build_metadata_filter(keywords):
    """Build a valid metadata filter for Chroma."""
    metadata_filter = {}
    
    # Only include valid keys for metadata filtering
    valid_keys = {"title", "president", "publication_date", "signing_date", "document_number", "executive_order_number"}
    for key, value in keywords.items():
        if key in valid_keys and value:
            # Ensure values are sanitized and simple strings
            metadata_filter[key] = value.strip()

    return metadata_filter


def hybrid_search_with_metadata(prompt, db):
    """Perform a hybrid search using metadata and vector similarity."""
    # Extract keywords
    keywords = extract_keywords(prompt)
    
    # Build metadata filter
    metadata_filter = build_metadata_filter(keywords)

    # Collect results from metadata filters
    metadata_results = []
    if metadata_filter:
        for field, value in metadata_filter.items():
            try:
                # Apply each filter individually
                filter_results = db.similarity_search_with_score("", filter={field: value})
                metadata_results.extend(filter_results)
            except Exception as e:
                print(f"Filter error for field '{field}': {e}")

    # Perform vector search
    vector_results = db.similarity_search_with_score(prompt)

    # Combine results
    combined_results = metadata_results + vector_results

    # Deduplicate and rerank
    seen_docs = set()
    unique_results = []
    for doc, score in combined_results:
        if doc.page_content not in seen_docs:
            seen_docs.add(doc.page_content)
            unique_results.append((doc, score))

    return sorted(unique_results, key=lambda x: x[1])


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
