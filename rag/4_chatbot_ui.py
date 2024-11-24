"""PART IV: Design UI for the chatbot."""

# Import Python packages
import os
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
MEMORY_WINDOW_SIZE = 20

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


# Functions to format the response
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
                        verbose=False,
                    )

                    # Establish a memory buffer
                    memory = ConversationBufferWindowMemory(
                        memory_key="chat_history",
                        output_key="answer",
                        return_messages=True,
                        window_size=MEMORY_WINDOW_SIZE,
                    )

                    # Create query chain
                    query_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        memory=memory,
                        retriever=retriever,
                        verbose=False,
                        return_source_documents=True,
                    )
                    
                    # Execute the query with detailed prompt
                    initial_response = query_chain({"question": prompt})
                    source_documents = initial_response.get("source_documents", [])
                    formatted_sources = format_sources(source_documents)
                      
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
                    
                    # Execute the query
                    final_response = query_chain({"question": detailed_prompt})
                    st.write("### Response")
                    st.write(final_response["answer"])
                    st.write("### Sources")
                    st.write(formatted_sources)

                    # Add the Save As button
                    file_content = f"Prompt: {prompt}\n\nResponse: {initial_response['answer']}\n\nSources:\n{formatted_sources}"
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



