import sys
import os
import uvicorn
from fastapi import FastAPI
from loguru import logger
from dotenv import load_dotenv
from src.ingestion import upload_pdfs
from src.retrieval import query_index

sys.path.append(r"E:\xainab_RAG")


print(upload_pdfs)


load_dotenv()
logger.add(sys.stdout, level="INFO")
app = FastAPI()

uploads_dir = "uploads"
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

app.post("/upload_pdfs")(upload_pdfs)
app.post("/query")(query_index)

if __name__ == "__main__":
    logger.info("Starting FastAPI app...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
