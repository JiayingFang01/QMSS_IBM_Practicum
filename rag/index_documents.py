"""Index source documents and persist in vector embedding database."""

# Import Python Packages
import os
import json
import uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Assuming the script is in 'QMSS_IBM_Practicum/rag' and the data is in 'QMSS_IBM_Practicum/data'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FOLDER = os.path.join(BASE_DIR, 'data')
JSON_FILE = os.path.join(DATASET_FOLDER, 'eos_final.json')
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, 'rag')


def main():
    print("Ingesting documents from JSON...")
    all_docs, doc_ids = ingest_docs_from_json(JSON_FILE)
    
    # Check if documents and IDs are correctly populated
    if len(all_docs) == 0 or len(doc_ids) == 0:
        raise ValueError("No documents or document IDs were generated.")
    
    if len(all_docs) != len(doc_ids):
        raise ValueError("The number of document IDs does not match the number of documents.")
    
    print(f"Successfully ingested {len(all_docs)} documents. Starting persistence...")
    db = generate_embed_index(all_docs, doc_ids)
    print("All documents have been persisted successfully.")


def ingest_docs_from_json(file_path):
    """Ingests and processes documents from a JSON file, combining all fields for indexing."""
    all_docs = []
    doc_ids = []

    print(f"Loading JSON file from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)

        # Ensure data is a list of documents
        if isinstance(data, list):
            for doc in data:
                process_document(doc, all_docs, doc_ids)
        else:
            print(f"Unexpected JSON format. Expecting a list of documents but got {type(data)}.")
    
    print(f"Ingested {len(all_docs)} documents from JSON.")
    return all_docs, doc_ids


def process_document(doc, all_docs, doc_ids):
    """Extracts relevant fields from the document and converts them into chunks."""
    # Extract all relevant fields and combine them into one text block for indexing
    title = doc.get('title', '')
    president = doc.get('president', {}).get('name', '')
    publication_date = doc.get('publication_date', '')
    signing_date = doc.get('signing_date', '')
    citation = doc.get('citation', '')
    document_number = doc.get('document_number', '')
    executive_order_number = doc.get('executive_order_number', '')
    pdf_url = doc.get('pdf_url', '')
    toc_subject = doc.get('toc_subject', '')
    disposition_notes = doc.get('disposition_notes', '')
    full_text = doc.get('cleaned_text', '')

    # Combine all fields into a single text block
    combined_text = f"""
    Title: {title}
    President: {president}
    Publication Date: {publication_date}
    Signing Date: {signing_date}
    Citation: {citation}
    Document Number: {document_number}
    Executive Order Number: {executive_order_number}
    PDF URL: {pdf_url}
    TOC Subject: {toc_subject}
    Disposition Notes: {disposition_notes}
    Full Text: {full_text}
    """
    
    # Append the combined text for later processing
    all_docs.append(Document(page_content=combined_text))

    # Generate and append a unique ID for this document
    doc_id = document_number or str(uuid.uuid4())  # Use document_number if available, else generate UUID

    # Ensure uniqueness of the document ID by appending a counter to duplicates
    if doc_id in doc_ids:
        counter = 1
        new_doc_id = f"{doc_id}_{counter}"
        while new_doc_id in doc_ids:
            counter += 1
            new_doc_id = f"{doc_id}_{counter}"
        doc_id = new_doc_id

    doc_ids.append(doc_id)


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


def create_index_chroma(docs, embeddings, persist_dir, doc_ids):
    """Creates a Chroma vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,  # Adjust the chunk size based on your needs
        chunk_overlap=200
    )
    
    # Track the new split chunks and their corresponding IDs
    splits = []
    split_doc_ids = []

    # Split documents into chunks
    for i, doc in enumerate(docs):
        doc_chunks = text_splitter.split_documents([doc])  # Split the document into chunks
        
        # Add the resulting chunks to the splits list
        splits.extend(doc_chunks)
        
        # Append a unique chunk identifier to each document ID to avoid duplicates
        for chunk_idx, _ in enumerate(doc_chunks):
            split_doc_ids.append(f"{doc_ids[i]}_chunk_{chunk_idx}")

    # Check if the number of document IDs matches the number of chunks
    if len(splits) != len(split_doc_ids):
        raise ValueError(f"Mismatch between split documents ({len(splits)}) and document IDs ({len(split_doc_ids)}).")

    # Create the Chroma vector store with embeddings and unique document IDs for each chunk
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
        ids=split_doc_ids  # Ensure that each chunk has a unique ID
    )

    return db


if __name__ == "__main__":
    main()
