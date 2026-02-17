import os
from typing import List
from fastapi import File, UploadFile
from llama_index.core import SimpleDirectoryReader
from src.utils import create_or_update_index
from loguru import logger

logger.info("Importing ingestion.py module")

uploads_dir = "uploads"


async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info(f"Received {len(files)} file(s) for upload.")

        for file in files:
            file_path = os.path.join(uploads_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            logger.info(f"File uploaded: {file.filename}")

        new_documents = SimpleDirectoryReader(uploads_dir).load_data()
        create_or_update_index(new_documents)

        logger.info("PDFs uploaded and index updated successfully.")
        return {"message": "PDFs uploaded and index updated successfully"}

    except Exception as e:
        logger.error(f"Error uploading PDFs: {str(e)}")
        return {"error": str(e)}
