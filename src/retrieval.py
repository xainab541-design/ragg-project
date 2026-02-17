from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os
from src.utils import create_or_update_index
from llama_index.core import SimpleDirectoryReader
from loguru import logger

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
logger.info("Embedding Model Initialized")

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
Settings.llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

documents = SimpleDirectoryReader(
    r"E:\xainab_RAG\uploads"
).load_data()
index = create_or_update_index(documents)


async def query_index(query: dict):
    try:
        question = query.get("question")
        logger.info(f"Received query: {question}")

        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(question)
        relevant_documents = [
            node.node.metadata["file_name"] for node in response.source_nodes
        ]
        response_data = {
            "response_from_llm": response.response,
            "relevant_documents": relevant_documents,
        }

        logger.info("Query processed successfully.")
        return response_data

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"error": str(e)}
