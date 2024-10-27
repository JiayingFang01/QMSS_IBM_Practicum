import os
import pprint
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

def main():
    # Streamlit UI
    st.title("Conversational LLM on Executive Orders")

    # Input fields for environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    prompt = st.text_area("Enter your question:")

    # Run button
    if st.button("Run Query"):
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
            st.warning("Please provide all required inputs.")


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
    main()
