from fastapi import APIRouter, UploadFile, File
from models import ChatRequest
from rag import upload_document, retrieve_context
from web_search import search_web
from llm import ask_llm
from database import save_chat

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    message = upload_document(file)

    return {"message": message}


@router.post("/chat")
def chat(request: ChatRequest):

    rag_context = retrieve_context(request.user_input)

    web_context = search_web(request.user_input)

    answer = ask_llm(request.user_input, rag_context, web_context)

    save_chat(request.user_input, answer)

    return {
        "question": request.user_input,
        "rag_context": rag_context,
        "web_context": web_context,
        "answer": answer
    }