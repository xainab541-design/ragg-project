from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384

index = faiss.IndexFlatL2(dimension)

documents = []

def upload_document(file):

    reader = PdfReader(file.file)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embeddings = model.encode(chunks)

    index.add(np.array(embeddings))

    documents.extend(chunks)

    return "Document uploaded and indexed"


def retrieve_context(query):

    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), k=3)

    context = ""

    for i in indices[0]:
        if i < len(documents):
            context += documents[i] + "\n"

    return context