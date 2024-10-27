"""Script for creating retrieval pipeline and invoking an LLM."""

import os
import pprint
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.vectorstores.pgvector import PGVector
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# Log full text sent to LLM
VERBOSE = False

# Details of persisted embedding store index
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Size of window for buffered window memory
MEMORY_WINDOW_SIZE = 10

def main(prompt):
    # Check which environment variables are set and use the appropriate LLM
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
        # One could add additional LLMs here
        raise EnvironmentError("No language model environment variables found.")

    # Establish a memory buffer for conversational continuity
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        window_size=MEMORY_WINDOW_SIZE,
    )

    # Put together all of the components into the full
    # chain with memory and retrieval-augmented generation
    query_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        verbose=VERBOSE,
        return_source_documents=True,
    )

    query_response = query_chain({"question": prompt})
    pprint.pprint(query_response)


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

if __name__ == "__main__":
    prompt = "tell me what EO 13773 is about?"
    main(prompt)
