"""User interface for seeing vector database matches."""

import os
import streamlit as st
import pprint
from dotenv import load_dotenv
import logging

# Load the environment variables from the .env file
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

ANSWER_ROLE = "Document Index"
FIRST_MESSAGE = "Enter text to find document matches."
QUESTION_ROLE = "Searcher"
PLACE_HOLDER = "Your message"


# Cached shared objects
@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, multi_process=False)
    return embeddings


@st.cache_resource
def get_embed_db():
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


# Function to save messages to session state
def save_message(role, content, sources=None):
    logger.info(f"message: {role} - '{content}'")
    msg = {"role": role, "content": content, "sources": sources}
    st.session_state["messages"].append(msg)
    return msg


# Function to write message to chat
def write_message(msg):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["sources"]:
            for doc in msg["sources"]:
                st.text(pprint.pformat(doc.metadata))
                st.write(doc.page_content)


# Initialize session state if it doesn't already exist
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    save_message(ANSWER_ROLE, FIRST_MESSAGE)

# Load embeddings and vector database
embeddings = load_embeddings()
db = get_embed_db()

# Display previous messages
for msg in st.session_state["messages"]:
    write_message(msg)

# Handle new user input
if prompt := st.chat_input(PLACE_HOLDER):
    msg = save_message(QUESTION_ROLE, prompt)
    write_message(msg)

    docs_scores = db.similarity_search_with_score(prompt)
    docs = []
    for doc, score in docs_scores:
        doc.metadata["similarity_score"] = score
        docs.append(doc)

    msg = save_message(ANSWER_ROLE, "Matching Documents", docs)
    write_message(msg)

st.title("Show Document Matches")
