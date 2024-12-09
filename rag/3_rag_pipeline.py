"""PART III: Create retrieval pipeline with LLM."""

# Import Python packages
import os
import re
import pprint
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# Load the environment variables from the .env file
load_dotenv()

# Log full text sent to LLM
VERBOSE = False

# Details of persisted embedding store index
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Size of window for buffered window memory
MEMORY_WINDOW_SIZE = 30

def main(prompt):
    """ Main function to execute the retrieval pipeline and interact with the LLM."""
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Access persisted embeddings and expose through langchain retriever
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = get_embed_db(embeddings)
    retriever = db.as_retriever()

    if openai_api_key:
        print("Using OpenAI's GPT-3.5-turbo for language model.")
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.5,
            verbose=VERBOSE,
        )
    else:
        raise EnvironmentError("No language model environment variables found.")

    # Establish a memory buffer for conversational continuity
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        window_size=MEMORY_WINDOW_SIZE,
    )

    # Metadata-based retrieval
    metadata_results = []
    keywords = extract_keywords(prompt)
    metadata_filter = build_metadata_filter(keywords)
    if metadata_filter:
        print(f"Performing metadata-based search with filter: {metadata_filter}")
        metadata_results = hybrid_search_with_metadata(prompt, db)

    # Vector-based retrieval
    print(f"Performing vector-based search for:\n{prompt}")
    vector_results = db.similarity_search_with_score(prompt)

    # Combine results
    combined_results = metadata_results + vector_results

    # Format combined results
    source_documents = [doc for doc, score in combined_results]
    
    # Limit source_documents to top 8 before formatting
    source_documents = source_documents[:8]  # Limit to top 8 sources
    formatted_sources = format_sources(source_documents)
    
    # Extract keywords from the prompt
    keywords = extract_keywords(prompt)
    
    # Construct a string of keywords for inclusion in the prompt
    keywords_string = "\n".join(
        f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in keywords.items() if value
        )
    
    if not keywords_string:
        keywords_string = "No specific keywords were extracted from the query."
    
    # Add structured prompt engineering for general use
    relevant_paragraphs = "\n\n".join([doc.page_content for doc in source_documents[:10]])  # Limit to top 10 most relevant documents
    
    if not relevant_paragraphs:
        relevant_paragraphs = "No highly relevant content retrieved. Please infer the key points based on related context."

    # Add structured prompt engineering
    detailed_prompt = (
    f"You are an expert researcher and assistant specializing in U.S. Executive Orders (EOs) and other official documents. "
    f"Based on the retrieved information below, summarize the key points relevant to the user's query. "
    f"Provide concise, clear answers while ensuring accuracy.\n\n"
    f"User Query:\n{prompt}\n\n"
    f"Extracted Keywords:\n{keywords_string}\n\n"
    f"Retrieved Content:\n{relevant_paragraphs}\n\n"
    f"If you cannot directly find the answer, follow these steps:\n\n"
    f"1. Attempt to infer the answer using related terms or synonyms. For example:\n"
    f"   - If the question mentions 'AI' but no direct information is available, look for terms like 'Technology,' 'Software,' or 'Innovation.'\n"
    f"   - If the question is about 'Climate Change' but not explicitly mentioned, search for terms like 'Environment,' 'Sustainability,' or 'Emissions Reduction.'\n"
    f"   - For queries about 'Data Privacy,' try finding references to 'Security,' 'Information Protection,' or 'Cybersecurity.'\n"
    f"   - If the question mentions 'Executive Orders,' treat alternate terms like 'EO,' 'Executive Orders documents,' or 'Presidential directives' as equivalent.\n\n"
    f"2. If inference is not possible, clarify that the answer could not be found, but offer context from related paragraphs "
    f"to provide useful information. For example:\n"
    f"   - 'The sources do not specifically mention AI, but they discuss advancements in technology that could include AI applications.'\n"
    f"   - 'No specific mention of climate change mitigation, but the EO emphasizes renewable energy initiatives, which indirectly address emissions.'\n"
    f"   - 'Data privacy is not explicitly mentioned, but Section III discusses frameworks for securing sensitive information.'\n"
    f"   - 'The question refers to Executive Orders documents, which are Presidential directives. The retrieved paragraphs describe federal guidelines relevant to the topic.'\n\n"
    f"Retrieved Sources:\n{formatted_sources}\n"
    )
    
    # Put together all components into the full chain with memory and retrieval-augmented generation
    query_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        verbose=VERBOSE,
        return_source_documents=True,
    )

    final_response = query_chain({"question": detailed_prompt})
    answer = final_response.get("answer", "No answer generated.")
    structured_sources = extract_sources_with_metadata(source_documents)
    
    # Print final results
    result = {
       "Answer": answer,
       "Sources": structured_sources,
       }
    
    pprint.pprint(result)


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
            combined_score = (weight_metadata * (10 if doc.page_content in metadata_docs else 100)) + (weight_vector * score)
            combined_results.append((doc, combined_score))

    # Sort results by combined score (lower is better)
    return sorted(combined_results, key=lambda x: x[1])  # Lower score = higher relevance


def extract_sources_with_metadata(source_documents):
    """Extract and format metadata and content from source documents."""
    sources = []
    for doc in source_documents:
        metadata = doc.metadata
        page_content = doc.page_content
        
        # Extract full text and metadata
        clean_full_text = extract_clean_full_text(page_content)

        source_entry = {
            "President": metadata.get("president"),
            "Title": metadata.get("title"),
            "EO number": metadata.get("executive_order_number"),
            "Publication Date": metadata.get("publication_date"),
            "Signing Date": metadata.get("signing_date"),
            "PDF URL": metadata.get("pdf_url"),
            "Relevant Paragraph": clean_full_text,
        }
        sources.append(source_entry)
    return sources


def extract_clean_full_text(page_content):
    """Extract only the full text from the document, removing metadata."""
    full_text_marker = "Full Text: "
    marker_index = page_content.find(full_text_marker)
    
    if marker_index != -1:
        # Extract everything after the marker
        return page_content[marker_index + len(full_text_marker):].strip()
    else:
        # If the marker is not found, return the original text
        return page_content.strip()
    

def format_sources(source_documents):
    """Formats retrieved documents for the LLM prompt, numbering the sources."""
    formatted = []
    for i, doc in enumerate(source_documents[:8], start=1):  # Limit to top 8 sources
        metadata = doc.metadata
        page_content = extract_clean_full_text(doc.page_content)
        
        # Manually construct the paragraph text with source numbering
        paragraph_text = (
            f"Source {i}:\n\n"
            f"Title: {metadata.get('title', 'Unknown Title')}\n\n"
            f"President: {metadata.get('president', 'Unknown President')}\n\n"
            f"EO Number: {metadata.get('executive_order_number', 'Unknown EO')}\n\n"
            f"Signing Date: {metadata.get('signing_date', 'Unknown Date')}\n\n"
            f"Publication Date: {metadata.get('publication_date', 'Unknown Date')}\n\n"
            f"PDF URL: {metadata.get('pdf_url', 'Unknown URL')}\n\n"
            f"Relevant Paragraph:\n{page_content}"
        )
        formatted.append(paragraph_text)
    return "\n\n".join(formatted)


def get_embed_db(embeddings):
    """Retrieves the Chroma vector store using the specified embedding model."""
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    if chroma_persist_dir:
        db = get_chroma_db(embeddings, chroma_persist_dir)
    else:
        raise EnvironmentError("No vector store environment variables found.")
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
    prompt = "How has the average text length of executive orders changed since 2000?"
    main(prompt)