"""PART IV: Design UI for the chatbot."""

# Import Python packages
import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables from the .env file
load_dotenv()

# Constants
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
MEMORY_WINDOW_SIZE = 30

# Log full text sent to LLM
VERBOSE = False

# Custom CSS for sidebar and main page styling
st.markdown(
    """
    <style>
    /* Sidebar button styling */
[data-testid="stSidebar"] .stButton button {
    background-color: #ffffff; 
    color: #333333; 
    padding: 12px 0; 
    width: 100%; 
    font-size: 16px;
    text-align: center;
    border-radius: 8px;
    border: none; 
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
    margin: 10px 0; 
    transition: background-color 0.3s ease, box-shadow 0.3s ease;
}

[data-testid="stSidebar"] .stButton button:hover {
    background-color: #f0f0f0;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); 
}

/* Sidebar title styling */
[data-testid="stSidebar"] .sidebar-title {
    font-size: 24px;
    font-weight: 700;
    color: #333333;
    margin-bottom: 20px;
    text-align: center; /* Center-align title */
}

    /* Main content button styling for Run Query */
    .stButton button {
        background-color: #71a4de;
        color: #ffffff;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .stButton button:hover {
        background-color: #2b6ec2;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
    }

    /* Main content styling with card-like appearance */
    .main .block-container {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    /* Title styling */
    .custom-title {
        font-size: 26px;
        font-weight: 700;
        color: #333333;
        margin-bottom: 15px;
    }

    /* Section header styling */
    .section-header {
        font-size: 22px;
        font-weight: 600;
        color: #444444;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #71a4de;
        padding-bottom: 5px;
    }
    
    .custom-download-button button {
        display: inline-block !important;
        background-color: #4CAF50 !important;
        color: white !important;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Retrieve the database
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


# Functions 
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


# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Introduction"

# Define page switch function
def set_page(page):
    st.session_state.page = page

# Main function
def main():
    # Sidebar navigation
    st.sidebar.title("Executive Orders Chatbot - QMSS & IBM")
    st.sidebar.button("Introduction", on_click=set_page, args=("Introduction",), key="sidebar_intro")
    st.sidebar.button("How-to Guides", on_click=set_page, args=("How-to Guides",), key="sidebar_guides")
    st.sidebar.button("Ask the Chatbot", on_click=set_page, args=("Ask the Chatbot",), key="sidebar_chatbot")

    # Display page content based on session state
    if st.session_state.page == "Introduction":
        st.markdown("<div class='custom-title'>Introduction</div>", unsafe_allow_html=True)
        st.write(
            "Welcome to the Conversational LLM on Executive Orders! Here you can interact with a chatbot to get information about executive orders."
        )
        st.write(
            "This tool uses LangChain's Conversational Retrieval Chain to provide accurate answers based on executive order documents."
        )
        st.write(
            "This project was conducted as part of the QMSS Practicum Course at Columbia University, in partnership with IBM."
        )
        st.markdown("<div class='custom-title'>Data</div>", unsafe_allow_html=True)
        st.write(
            "Executive Orders Dataset link: https://www.federalregister.gov/presidential-documents/executive-orders"
        )
        st.write(
            "This link takes you to the Federal Register's official page for Executive Orders, which are legally binding directives from the President that guide the operations of the federal government. The page provides access to the full text of these orders, organized by date or topic, offering a clear view of the policies and priorities of different administrations."
        )
        st.write(
            "Our dataset focuses on executive orders issued by the most recent five presidents."
        )
        st.markdown("<div class='custom-title'>Github</div>", unsafe_allow_html=True)
        st.write(
            "GitHub link: https://github.com/JiayingFang01/QMSS_IBM_Practicum"
        )
        st.write(
        " Please see the GitHub repository for the code and project details."
        )
    elif st.session_state.page == "How-to Guides":
        st.markdown("<div class='custom-title'>How-to Guides</div>", unsafe_allow_html=True)
        st.write("Learn how to use the chatbot and interact with executive order documents.")
        st.markdown("<div class='section-header'>Step 1: Ask a Question</div>", unsafe_allow_html=True)
        st.write("Type your question about an executive order in the input box.")
        st.markdown("<div class='section-header'>Step 2: Get Response</div>", unsafe_allow_html=True)
        st.write("Click 'Run Query' to get an answer.")
        st.markdown("<div class='section-header'>FAQ💡</div>", unsafe_allow_html=True)
        with st.expander("Why are executive orders important?"):
            st.write("""
            Executive orders allow the President to take swift action on critical issues
             without waiting for Congress to pass legislation.
            """)
        with st.expander("Which executive orders are included in this chatbot?"):
            st.write("""
            The chatbot includes executive orders since 1937, 
            spanning topics like national security, public health, and economic policy
            """)
        with st.expander("How can I search for a specific executive order?"):
            st.write("""
            You can search by keywords,
            the President’s name, the signing date, or the topic of the executive order.
            """)
        with st.expander("Can the chatbot explain the purpose of an executive order?"):
            st.write("""
            Yes, the chatbot can provide an overview of the purpose and
             context of specific executive orders.
            """)
        with st.expander("What should I do if the chatbot cannot answer my question?"):
            st.write("""
                   If the chatbot cannot provide an answer,
                    try re-running the query or rephrasing your question. 
                    If it still cannot work, 
                    please visit the Federal Register's official website for more detailed information.
                   """)

    elif st.session_state.page == "Ask the Chatbot":
        st.markdown("<div class='custom-title'>Chatbot 💬</div>", unsafe_allow_html=True)
        prompt = st.text_area("How Can I Help You?")

        # Run button and logic
        if st.button("Run Query", key="main_run_query"):
            openai_api_key = os.getenv("OPENAI_API_KEY")
            chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
            if openai_api_key and chroma_persist_dir and prompt:
                try:
                    # Set environment variables for Chroma
                    os.environ["OPENAI_API_KEY"] = openai_api_key
                    os.environ["CHROMA_PERSIST_DIR"] = chroma_persist_dir

                    # Load embedding model
                    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                    db = get_embed_db(embeddings)
                    retriever = db.as_retriever()
                    
                    # Load LLM
                    llm = ChatOpenAI(
                        openai_api_key=openai_api_key,
                        model="gpt-3.5-turbo",
                        temperature=0.5,
                        verbose=VERBOSE,
                        )
                    
                    # Establish a memory buffer
                    memory = ConversationBufferWindowMemory(
                        memory_key="chat_history",
                        output_key="answer",
                        return_messages=True,
                        window_size=MEMORY_WINDOW_SIZE,
                        )
                    
                    # Metadata-based retrieval
                    keywords = extract_keywords(prompt)
                    metadata_filter = build_metadata_filter(keywords)
                    metadata_results = []
                    if metadata_filter:
                        metadata_results = hybrid_search_with_metadata(prompt, db)
                    
                    # Vector-based retrieval
                    vector_results = db.similarity_search_with_score(prompt)

                    # Combine metadata and vector results
                    combined_results = metadata_results + vector_results
                    source_documents = [doc for doc, score in combined_results]

                    # Limit sources to top 8 for efficiency
                    source_documents = source_documents[:8]
                    formatted_sources = format_sources(source_documents)
                    
                    # Extract keywords from the prompt
                    keywords = extract_keywords(prompt)

                    # Prepare keywords string
                    keywords_string = "\n".join(
                        f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in keywords.items() if value
                        ) 
                    
                    if not keywords_string:
                        keywords_string = "No specific keywords were extracted from the query."

                    # Prepare relevant paragraphs
                    relevant_paragraphs = "\n\n".join([doc.page_content for doc in source_documents])
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
                    
                    # Create query chain
                    query_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        memory=memory,
                        retriever=retriever,
                        verbose=VERBOSE,
                        return_source_documents=True,
                    )
                    
                    # Execute the query with the detailed prompt
                    final_response = query_chain({"question": detailed_prompt})
                    answer = final_response.get("answer", "No answer generated.")
                    st.write("### Response")
                    st.write(answer)

                    # Show retrieved sources
                    st.write("### Sources") 
                    st.write(formatted_sources)

                    # Add the Save As button
                    file_content = f"Prompt: {prompt}\n\nResponse: {answer}\n\nSources:\n{formatted_sources}"
                    st.download_button(
                        label="Download Your Response",
                        data=file_content,
                        file_name="IBM_Model_Response.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please provide all required inputs in the environment.")

if __name__ == "__main__":
    main()



