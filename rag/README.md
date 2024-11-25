# RAG
Before running the code, ensure the [.env](https://github.com/JiayingFang01/QMSS_IBM_Practicum/blob/main/rag/.env) file is in the folder on your local computer and that your API key is added to it.

## PART 1: Document and Query Embeddings
1. Ingest and Process Documents:
- Extract metadata, full text, and additional information from the JSON file.
2. Chunking:
- Chunks are created using sentence_based_chunking, with the following configurable parameters:
  - CHUNK_SIZE: Maximum character length for each chunk (default: 1200).
  - CHUNK_OVERLAP: Number of overlapping characters between consecutive chunks (default: 200).
- Metadata is preserved for each chunk to maintain context during retrieval.
3. Generate Embeddings:
- Use the HuggingFace model (sentence-transformers/multi-qa-mpnet-base-dot-v1) to generate embeddings for document chunks.
- Each chunk is stored with its corresponding embedding, metadata, and a unique identifier.
4. Persist to Chroma:
- Save the embeddings, metadata, and chunk information into a persistent Chroma vector database.


## PART 2: Embedding-Based Retrieval
1. Hybrid Search:
- Combines metadata filtering (exact matches on fields like title, president, publication date, signing date, EO number, etc.) with vector similarity to retrieve the most relevant results.
2. Metadata Extraction:
- Extract keywords from the query to filter documents based on specific fields like title, executive order number, or president.
3. Vector Similarity Search:
- Embedding-based similarity search retrieves documents matching the query contextually.
4. Rerank and Deduplicate Results:
- Combine results from metadata and vector searches, ensuring no duplicates.
