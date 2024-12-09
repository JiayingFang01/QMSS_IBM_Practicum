# RAG
Before running the code, ensure the .env file is in the folder on your local computer and that your API key is added to it.

## How to Use
- Step 0: Make sure you have [.env](https://github.com/JiayingFang01/QMSS_IBM_Practicum/blob/main/rag/env.default) file in the folder on your local computer. Please **add your own API key** in it. 
- Step 1 : Create the vecter database. Please run [1_document_indexer.py](https://github.com/JiayingFang01/QMSS_IBM_Practicum/blob/main/rag/1_document_indexer.py).  
- Step 2: Run [2_embedding_retrieval.py](https://github.com/JiayingFang01/QMSS_IBM_Practicum/blob/main/rag/2_embedding_retrieval.py). You can modify the sample query to test the retrieval process. 
- Step 3: Run [3_rag_pipeline.py](https://github.com/JiayingFang01/QMSS_IBM_Practicum/blob/main/rag/3_rag_pipeline.py). You can modify the query to test it.
- Step 4: Run [4_chatbot_ui.py](https://github.com/JiayingFang01/QMSS_IBM_Practicum/blob/main/rag/4_chatbot_ui.py). You can run it in Terminal using the command "Streamlit run 4_chatbot_ui.py".

## PART 1: Document and Query Embeddings
**1. Ingest and Process Documents:**
- Extract metadata, full text, and additional information from the JSON file.
  
**2. Chunking:**
- Chunks are created using sentence_based_chunking, with the following configurable parameters:
  - CHUNK_SIZE: Maximum character length for each chunk (default: 1200).
  - CHUNK_OVERLAP: Number of overlapping characters between consecutive chunks (default: 200).
- Metadata is preserved for each chunk to maintain context during retrieval.

**3. Generate Embeddings:**
- Use the HuggingFace model (sentence-transformers/multi-qa-mpnet-base-dot-v1) to generate embeddings for document chunks.
- Each chunk is stored with its corresponding embedding, metadata, and a unique identifier.

**4. Persist to Chroma:**
- Save the embeddings, metadata, and chunk information into a persistent Chroma vector database.


## PART 2: Embedding-Based Retrieval
**1. Hybrid Search:**
- Combines metadata filtering (exact matches on fields like title, president, publication date, signing date, EO number, etc.) with vector similarity to retrieve the most relevant results.

**2. Metadata Extraction:**
- Extract keywords from the query to filter documents based on specific fields like title, executive order number, or president.
  
**3. Vector Similarity Search:**
- Embedding-based similarity search retrieves documents matching the query contextually.
  
**4. Rerank and Deduplicate Results:**
- Combine results from metadata and vector searches, ensuring no duplicates.

## PART 3: Retrieval Pipeline with LLM
**1. LLM Setup:**
- Model: GPT-3.5-turbo
- Memory Buffer: Implements ConversationBufferWindowMemory for contextual continuity with a window size of 30 turns.
    
**2. Data Retrieval:**
- Metadata Search: Extract keywords (EO numbers, presidents, dates, etc.) and apply filters for document matching.
- Vector Search: Perform similarity-based retrieval using embeddings.

## PART 4: UI Design
- UI Design: Rely on Streamlit to design the UI.
- Testing link: https://jiayingfang01-qmss-ibm-practicum-rag4-chatbot-ui-ousaoc.streamlit.app/


