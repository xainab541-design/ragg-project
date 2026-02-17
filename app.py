import os
import sys
import json
import uuid
import logging
import qdrant_client
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

load_dotenv()
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
Settings.llm = Gemini(
    model="models/gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader(
    r"E:\xainab_RAG\uploads"
).load_data()

client = qdrant_client.QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="paul_graham2")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
response = query_engine.query("what is speciality of decoder based models?")


print("=========================Response=========================")
response.print_response_stream()
print("=========================Response=========================")
