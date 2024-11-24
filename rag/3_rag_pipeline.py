"""PART III: Create retrieval pipeline with LLM."""

# Import Python packages
import os
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
MEMORY_WINDOW_SIZE = 20

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

    # Put together all components into the full chain with memory and retrieval-augmented generation
    query_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        verbose=VERBOSE,
        return_source_documents=True,
    )

    # Query the chain
    query_response = query_chain({"question": prompt})
    
    # Retrieve the relevant source documents
    source_documents = query_response.get("source_documents", [])
    formatted_sources = format_sources(source_documents)

    # Add structured prompt engineering
    detailed_prompt = (
    f"You are an expert on U.S. Executive Orders (EOs). Based on the retrieved information below, "
    f"answer the user's question concisely, and include the specific paragraph(s) used. "
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

    # Run the refined prompt through the query chain
    final_response = query_chain({"question": detailed_prompt})
    answer = final_response.get("answer", "No answer generated.")
    structured_sources = extract_sources_with_metadata(source_documents)
    
    # Print final results
    result = {
       "Answer": answer,
       "Sources": structured_sources,
       }
    
    pprint.pprint(result)


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
    for i, doc in enumerate(source_documents, start=1):
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
    prompt = "Please give me 3 EOs related to public health. Include the president name and signing date."
    main(prompt)
