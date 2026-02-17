import json
import uuid
import logging
import sys
import os
import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SimpleNodeParser

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is required. Set it in your environment or .env file.")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    embed_batch_size=10
)

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
Settings.llm = Gemini(
    model="models/gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

uploads_dir = "uploads"
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

client = qdrant_client.QdrantClient(host="localhost", port=6333)

vector_store = QdrantVectorStore(client=client, collection_name="xainab_collection")


def load_documents_from_uploads():
    if not os.listdir(uploads_dir):
        return []
    documents = SimpleDirectoryReader(uploads_dir).load_data()
    
    # Filter out documents with empty or invalid text content
    valid_documents = []
    for doc in documents:
        try:
            # Get text content and ensure it's a valid string
            text = doc.get_content() if hasattr(doc, 'get_content') else (doc.text if hasattr(doc, 'text') else str(doc))
            
            # Filter out empty, None, or non-string content
            if text and isinstance(text, str) and text.strip() and len(text.strip()) > 0:
                # Document has valid text, include it
                valid_documents.append(doc)
        except Exception as e:
            # Log but skip problematic documents
            logging.warning(f"Skipping document during loading due to error: {e}")
            continue
    
    return valid_documents


def create_or_update_index(documents):
    if not documents:
        return None
    
    # Additional validation before indexing
    filtered_docs = []
    for doc in documents:
        try:
            text = doc.get_content() if hasattr(doc, 'get_content') else (doc.text if hasattr(doc, 'text') else str(doc))
            # Ensure text is valid string and not empty
            if text and isinstance(text, str) and text.strip() and len(text.strip()) > 0:
                # Document has valid text, include it (don't try to modify doc.text as it's read-only)
                filtered_docs.append(doc)
        except Exception as e:
            # Log but skip problematic documents
            logging.warning(f"Skipping document due to error: {e}")
            continue
    
    if not filtered_docs:
        logging.warning("No valid documents to index after filtering")
        return None
    
    # Create node parser
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=1024,
        chunk_overlap=20,
        include_metadata=True
    )
    
    # Parse documents into nodes first
    all_nodes = []
    for doc in filtered_docs:
        try:
            nodes = node_parser.get_nodes_from_documents([doc])
            # Filter out nodes with empty or invalid text
            for node in nodes:
                try:
                    node_text = node.get_content() if hasattr(node, 'get_content') else (node.text if hasattr(node, 'text') else str(node))
                    
                    # Ensure node text is valid string
                    if not isinstance(node_text, str):
                        node_text = str(node_text) if node_text is not None else ""
                    
                    # Remove any null bytes or control characters that might cause issues
                    node_text = node_text.replace('\x00', '').replace('\ufffd', '')
                    
                    # Clean whitespace
                    node_text = ' '.join(node_text.split())  # Normalize whitespace
                    
                    # Ensure text is not empty after cleaning
                    if not node_text or len(node_text.strip()) == 0:
                        logging.debug(f"Skipping node with empty text after cleaning")
                        continue
                    
                    # Ensure text has minimum length (at least 10 characters)
                    if len(node_text.strip()) < 10:
                        logging.debug(f"Skipping node with text too short: {len(node_text)} chars")
                        continue
                    
                    # Ensure text contains at least some alphanumeric characters
                    if not any(c.isalnum() for c in node_text):
                        logging.debug(f"Skipping node with no alphanumeric characters")
                        continue
                    
                    # Update node text with cleaned version (if possible)
                    # Note: We can't always modify node.text, so we'll just validate
                    # The actual text will be used during embedding
                    
                    all_nodes.append(node)
                        
                except Exception as e:
                    logging.debug(f"Skipping node due to error: {e}")
                    continue
        except Exception as e:
            logging.warning(f"Error parsing document into nodes: {e}")
            continue
    
    if not all_nodes:
        logging.warning("No valid nodes to index after filtering")
        return None
    
    if not all_nodes:
        logging.warning("No valid nodes to index after filtering")
        return None
    
    logging.info(f"Created {len(all_nodes)} valid nodes from {len(filtered_docs)} documents")
    
    # Pre-embed nodes individually to filter out any that fail embedding
    embed_model = Settings.embed_model
    valid_embedded_nodes = []
    failed_count = 0
    
    for node in all_nodes:
        try:
            node_text = node.get_content() if hasattr(node, 'get_content') else (node.text if hasattr(node, 'text') else str(node))
            
            # Ensure text is a valid string
            if not isinstance(node_text, str):
                node_text = str(node_text) if node_text is not None else ""
            
            # Clean text
            node_text = node_text.replace('\x00', '').replace('\ufffd', '')
            node_text = ' '.join(node_text.split())
            
            # Skip if empty after cleaning
            if not node_text or len(node_text.strip()) < 10:
                failed_count += 1
                continue
            
            # Try to generate embedding
            try:
                embedding = embed_model.get_text_embedding(node_text)
                if embedding and len(embedding) > 0:
                    # Set embedding on node
                    node.embedding = embedding
                    valid_embedded_nodes.append(node)
                else:
                    failed_count += 1
                    logging.debug(f"Node embedding generation returned empty")
            except Exception as embed_error:
                failed_count += 1
                logging.debug(f"Node embedding failed: {embed_error}")
                continue
                
        except Exception as e:
            failed_count += 1
            logging.debug(f"Error processing node: {e}")
            continue
    
    if failed_count > 0:
        logging.info(f"Filtered out {failed_count} nodes that failed embedding")
    
    if not valid_embedded_nodes:
        logging.warning("No nodes with valid embeddings after processing")
        return None
    
    logging.info(f"Successfully embedded {len(valid_embedded_nodes)} nodes")
    
    # Create index from pre-embedded nodes
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=valid_embedded_nodes,
        storage_context=storage_context
    )
    return index


def build_chat_engine(current_index):
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
    return current_index.as_chat_engine(chat_mode="context", memory=memory)


if "index" not in st.session_state:
    initial_docs = load_documents_from_uploads()
    st.session_state["index"] = (
        create_or_update_index(initial_docs) if initial_docs else None
    )
    st.session_state["chat_engine"] = (
        build_chat_engine(st.session_state["index"])
        if st.session_state["index"]
        else None
    )
    st.session_state["chat_history"] = []
else:
    if "chat_engine" not in st.session_state:
        st.session_state["chat_engine"] = (
            build_chat_engine(st.session_state["index"])
            if st.session_state["index"]
            else None
        )
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

index = st.session_state["index"]
st.title("PDF Vector Search Application")
st.sidebar.header("PDF Ingestion")
pdf_uploaded = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)
if st.sidebar.button("Clear chat history"):
    st.session_state["chat_history"] = []
    st.session_state["chat_engine"] = build_chat_engine(index) if index else None
    st.sidebar.success("Chat history cleared.")

if pdf_uploaded:
    st.sidebar.write("PDFs uploaded successfully.")

    for uploaded_file in pdf_uploaded:
        try:
            file_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["uploaded_files"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error saving file: {uploaded_file.name} - {e}")

    try:
        updated_documents = load_documents_from_uploads()
        if updated_documents:
            index = create_or_update_index(updated_documents)
            st.session_state["index"] = index
            st.session_state["chat_engine"] = (
                build_chat_engine(index) if index else None
            )
            st.session_state["chat_history"] = []
            st.sidebar.write("Index updated!")
        else:
            st.sidebar.warning("No documents found to index.")
    except Exception as e:
        st.error(f"Error updating index: {e}")

st.header("Chat with your documents")

chat_history = st.session_state.get("chat_history", [])
for message in chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Ask anything about your PDFs")
if user_prompt:
    if index is None:
        st.warning("No documents indexed yet. Please upload PDFs first.")
    else:
        st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        if st.session_state.get("chat_engine") is None:
            st.session_state["chat_engine"] = build_chat_engine(index)

        chat_engine = st.session_state["chat_engine"]
        chat_response = chat_engine.chat(user_prompt)
        assistant_reply = getattr(chat_response, "response", None) or chat_response.message.content

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": assistant_reply}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
