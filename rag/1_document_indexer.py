"""PART I: Create a vector database."""

# Import Python packages
import os
import json
import uuid
import nltk
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download NLTK's sentence tokenizer
nltk.download("punkt")

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FOLDER = os.path.join(BASE_DIR, 'data')
JSON_FILE = os.path.join(DATASET_FOLDER, 'eos_final.json')
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(BASE_DIR, 'rag'))

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200


def main():
    """Main function for processing and indexing documents."""
    print("Ingesting documents from JSON...")
    all_docs, doc_ids = ingest_docs_from_json(JSON_FILE)
    
    # Validate ingested data
    if len(all_docs) == 0 or len(doc_ids) == 0:
        raise ValueError("No documents or document IDs were generated.")
    if len(all_docs) != len(doc_ids):
        raise ValueError("The number of document IDs does not match the number of documents.")

    print(f"Successfully ingested {len(all_docs)} documents. Starting persistence...")
    db = generate_embed_index(all_docs, doc_ids)
    print("All documents have been persisted successfully.")


def ingest_docs_from_json(file_path):
    """Ingests and processes documents from a JSON file."""
    all_docs = []
    doc_ids = []

    print(f"Loading JSON file from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)

        if isinstance(data, list):
            for doc in data:
                process_document(doc, all_docs, doc_ids)
        else:
            print(f"Unexpected JSON format. Expecting a list of documents but got {type(data)}.")

    print(f"Ingested {len(all_docs)} documents from JSON.")
    return all_docs, doc_ids


def process_document(doc, all_docs, doc_ids):
    """Processes a document into chunks with metadata."""
    def sanitize_metadata(value):
        """Convert None values to empty strings or valid default values."""
        return value if value is not None else ""

    metadata = {
        "title": sanitize_metadata(doc.get('title')),
        "president": sanitize_metadata(doc.get('president')),
        "publication_date": sanitize_metadata(doc.get('publication_date')),
        "signing_date": sanitize_metadata(doc.get('signing_date')),
        "citation": sanitize_metadata(doc.get('citation')),
        "document_number": sanitize_metadata(doc.get('document_number')),
        "executive_order_number": sanitize_metadata(doc.get('executive_order_number')),
        "pdf_url": sanitize_metadata(doc.get('pdf_url')),
        "toc_subject": sanitize_metadata(doc.get('toc_subject')),
        "disposition_notes": sanitize_metadata(doc.get('disposition_notes')),
    }
    full_text = doc.get('full_text', '')

    combined_text = f"""
    Title: {metadata["title"]}
    President: {metadata["president"]}
    Publication Date: {metadata["publication_date"]}
    Signing Date: {metadata["signing_date"]}
    Citation: {metadata["citation"]}
    Document Number: {metadata["document_number"]}
    Executive Order Number: {metadata["executive_order_number"]}
    PDF URL: {metadata["pdf_url"]}
    Topic: {metadata["toc_subject"]}
    Disposition Notes: {metadata["disposition_notes"]}
    Full Text: {full_text}
    """

    all_docs.append(Document(page_content=combined_text, metadata=metadata))

    doc_id = metadata["document_number"] or str(uuid.uuid4())
    if doc_id in doc_ids:
        counter = 1
        while f"{doc_id}_{counter}" in doc_ids:
            counter += 1
        doc_id = f"{doc_id}_{counter}"
    doc_ids.append(doc_id)


def sentence_based_chunking(doc, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Splits a document into sentence-preserving chunks."""
    text = doc.page_content
    metadata = doc.metadata

    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size:
            chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata))
            current_chunk = sentence
            current_length = sentence_length
        else:
            current_chunk += " " + sentence
            current_length += sentence_length

    if current_chunk:
        chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata))

    if chunk_overlap > 0:
        overlapped_chunks = []
        for i in range(len(chunks)):
            overlapped_chunks.append(chunks[i])
            if i > 0:
                overlap_text = chunks[i - 1].page_content[-chunk_overlap:] + " " + chunks[i].page_content
                overlapped_chunks[-1] = Document(page_content=overlap_text.strip(), metadata=metadata)
        chunks = overlapped_chunks

    return chunks


def generate_embed_index(docs, doc_ids):
    """Generates an embedding index for the documents."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")

    if chroma_persist_dir:
        print(f"Persisting embeddings to Chroma at {chroma_persist_dir}...")
        db = create_index_chroma(docs, embeddings, chroma_persist_dir, doc_ids)
        print("Embeddings persisted.")
    else:
        raise EnvironmentError("No vector store environment variables found.")
    
    return db


def validate_metadata(metadata):
    """Ensure all metadata values are valid."""
    return {key: (value if value is not None else "") for key, value in metadata.items()}


def create_index_chroma(docs, embeddings, persist_dir, doc_ids):
    """Creates a Chroma vector store with sentence-preserving chunks and metadata."""
    splits = []
    split_doc_ids = []
    split_metadata = []

    for i, doc in enumerate(docs):
        doc_chunks = sentence_based_chunking(doc)

        for chunk_idx, chunk in enumerate(doc_chunks):
            splits.append(chunk.page_content)  # Extract text content for embedding
            split_doc_ids.append(f"{doc_ids[i]}_chunk_{chunk_idx}")
            chunk_metadata = validate_metadata(chunk.metadata.copy())  # Validate metadata
            chunk_metadata["chunk_id"] = f"{doc_ids[i]}_chunk_{chunk_idx}"
            split_metadata.append(chunk_metadata)

    if len(splits) != len(split_metadata):
        raise ValueError("Mismatch between document chunks and metadata.")

    # Create the Chroma vector store
    db = Chroma.from_texts(
        texts=splits, 
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
        ids=split_doc_ids,  # Unique IDs for each chunk
        metadatas=split_metadata,  # Corresponding metadata for each chunk
    )

    return db


if __name__ == "__main__":
    main()
