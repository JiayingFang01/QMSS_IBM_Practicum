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
MEMORY_WINDOW_SIZE = 10

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
    </style>
    """,
    unsafe_allow_html=True
)

# Helper functions
def get_embed_db(embeddings):
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    if chroma_persist_dir:
        db = get_chroma_db(embeddings, chroma_persist_dir)
    else:
        raise EnvironmentError("No vector store environment variables found.")
    return db

def get_chroma_db(embeddings, persist_dir):
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return db

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Introduction"

# Define page switch function
def set_page(page):
    st.session_state.page = page

# Main function
def main():
    # Sidebar navigation
    st.sidebar.title("Executive Orders Chatbot-IBM")
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

    elif st.session_state.page == "How-to Guides":
        st.markdown("<div class='custom-title'>How-to Guides</div>", unsafe_allow_html=True)
        st.write("Learn how to use the chatbot and interact with executive order documents.")
        st.markdown("<div class='section-header'>Step 1: Ask a Question</div>", unsafe_allow_html=True)
        st.write("Type your question about an executive order in the input box.")
        st.markdown("<div class='section-header'>Step 2: Get Response</div>", unsafe_allow_html=True)
        st.write("Click 'Run Query' to get an answer.")

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

                    # Execute the query
                    query_response = query_chain({"question": prompt})
                    st.write("### Response")
                    st.write(query_response["answer"])

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please provide all required inputs in the environment.")

if __name__ == "__main__":
    main()













