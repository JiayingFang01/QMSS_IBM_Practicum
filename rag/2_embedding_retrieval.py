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
    What are the key points of Executive Order 13773
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
    """Extract keywords for metadata fields from the query, including multiple occurrences."""
    keywords = {}

    # Extract all Executive Order numbers or similar terms
    eo_matches = re.findall(r"(EO|Executive Order|Presidential Order|Order)\s*(\d+)", prompt, re.IGNORECASE)
    if eo_matches:
        keywords["executive_order_number"] = [match[1] for match in eo_matches]

    # Extract all President names
    president_matches = re.findall(
        r"President\s+(Joseph\s+Biden|Barack\s+Obama|Donald\s+Trump|George\s+(W\s+)?Bush|Bill\s+Clinton|Ronald\s+Reagan)",
        prompt, re.IGNORECASE
    )
    if president_matches:
        # Convert tuples to simple strings
        keywords["president"] = list(set(match[0] if isinstance(match, tuple) else match for match in president_matches))

    # Extract all Publication Dates or similar terms
    publication_date_matches = re.findall(
        r"(?:published|released|effective|publication date|date of issuance)\s+(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2})",
        prompt, re.IGNORECASE
    )
    if publication_date_matches:
        keywords["publication_date"] = list(set(publication_date_matches))

    # Extract all Signing Dates or similar terms
    signing_date_matches = re.findall(
        r"(?:signed on|signing date|issued on)\s+(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2})",
        prompt, re.IGNORECASE
    )
    if signing_date_matches:
        keywords["signing_date"] = list(set(signing_date_matches))

    # Infer Years from Dates
    year_matches = re.findall(r"\b(19|20)\d{2}\b", prompt)
    if year_matches:
        keywords["year"] = list(set(year_matches))

    # Extract all Document Numbers or similar terms
    document_number_matches = re.findall(r"(Document Number|Doc No|#)\s+(\d+)", prompt, re.IGNORECASE)
    if document_number_matches:
        keywords["document_number"] = [match[1] for match in document_number_matches]

    # Extract all Titles (General Topics or Subjects)
    title_matches = re.findall(r"(on|about|regarding)\s+([\w\s]+?)(?:,|\s+and|\s*$)", prompt, re.IGNORECASE)
    if title_matches:
        keywords["title"] = [match[1].strip() for match in title_matches]

    return keywords


def build_metadata_filter(keywords):
    """Build a valid metadata filter for Chroma."""
    metadata_filter = {}
    
    # Include valid keys for metadata filtering
    valid_keys = {"title", "president", "publication_date", "signing_date", "document_number", "executive_order_number"}
    for key, value in keywords.items():
        if key in valid_keys and value:
            if isinstance(value, list):
                # Handle multiple values using OR-like logic
                metadata_filter[key] = {"$in": value}
            else:
                metadata_filter[key] = value.strip() if isinstance(value, str) else value

    return metadata_filter


def hybrid_search_with_metadata(prompt, db):
    """Perform a hybrid search using metadata and vector similarity."""
    # Extract keywords
    keywords = extract_keywords(prompt)

    # Collect results from metadata filters
    metadata_results = []
    for key, values in keywords.items():
        if isinstance(values, list):  # Handle multiple values for a metadata field
            for value in values:
                try:
                    filter_results = db.similarity_search_with_score("", filter={key: value})
                    # Assign a low score (e.g., 10) for metadata matches
                    metadata_results.extend((doc, 10) for doc, _ in filter_results)
                except Exception as e:
                    print(f"Filter error for field '{key}' with value '{value}': {e}")
        else:  # Handle single value
            try:
                filter_results = db.similarity_search_with_score("", filter={key: values})
                # Assign a low score (e.g., 10) for metadata matches
                metadata_results.extend((doc, 10) for doc, _ in filter_results)
            except Exception as e:
                print(f"Filter error for field '{key}' with value '{values}': {e}")

    # Perform vector search
    vector_results = db.similarity_search_with_score(prompt)

    # Combine results with weighted scores
    combined_results = []
    seen_docs = set()

    # Use document content as a unique identifier
    metadata_docs = {doc.page_content for doc, _ in metadata_results}

    for doc, score in metadata_results + vector_results:
        doc_id = id(doc)  # Use id() or doc.page_content as a unique identifier
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            # Combine scores with weighted adjustment
            weight_metadata = 0.5  # Adjust as needed
            weight_vector = 0.5    # Adjust as needed
            # Metadata score is 10 if matched, 100 otherwise
            combined_score = (weight_metadata * (10 if doc.page_content in metadata_docs else 50)) + (weight_vector * score)
            combined_results.append((doc, combined_score))

    # Sort results by combined score (lower is better)
    return sorted(combined_results, key=lambda x: x[1])  # Lower score = higher relevance


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
