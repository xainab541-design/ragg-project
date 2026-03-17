# 🤖 AI Chatbot Project (FastAPI + MongoDB + RAG)

## 📌 Overview

This project is a **web-based AI chatbot application** that enables real-time interaction between users and an AI system. It combines:

* ⚡ FastAPI (Backend)
* 🧠 LLM via OpenRouter
* 🌐 Tavily (Web Search)
* 📄 RAG (PDF-based Retrieval)
* 🍃 MongoDB (Database)

The chatbot can:

* Answer user queries
* Retrieve context from uploaded PDFs
* Fetch relevant web data
* Store chat history in MongoDB

---

## 🏗️ Project Structure

```
AI_Chatbot_Project
│
├── main.py            # Entry point
├── config.py          # Environment variables
├── database.py        # MongoDB connection
├── models.py          # Request schemas
├── rag.py             # PDF + embeddings + retrieval
├── web_search.py      # Tavily search
├── llm.py             # LLM API integration
├── routes.py          # API endpoints
│
├── requirements.txt
└── .env
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
cd AI_Chatbot_Project
```

### 2️⃣ Create virtual environment

```
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 🔐 Environment Variables (.env)

Create a `.env` file and add:

```
OPENROUTER_API_KEY=your_api_key
TAVILY_API_KEY=your_api_key
```

---

## 🍃 MongoDB Setup

Make sure MongoDB is running locally:

```
mongodb://localhost:27017
```

Database: `chatbot_db`
Collection: `chats`

---

## ▶️ Running the Application

```
uvicorn main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

### 1️⃣ Upload PDF (RAG)

**POST** `/upload`

* Upload a PDF file
* Extracts text and stores embeddings

---

### 2️⃣ Chat Endpoint

**POST** `/chat`

#### Request Body:

```
{
  "user_input": "What is AI?"
}
```

#### Response:

```
{
  "question": "What is AI?",
  "rag_context": "...",
  "web_context": "...",
  "answer": "AI stands for Artificial Intelligence..."
}
```

---

## 🧠 Features

* ✅ Real-time chatbot interaction
* ✅ Retrieval-Augmented Generation (RAG)
* ✅ PDF document understanding
* ✅ Web search integration
* ✅ MongoDB chat storage
* ✅ Modular clean architecture

---

## 🔄 Workflow

1. User sends a question
2. System retrieves:

   * PDF context (RAG)
   * Web results (Tavily)
3. LLM generates response
4. Chat is stored in MongoDB

---

## 🛠️ Tech Stack

* FastAPI
* MongoDB (PyMongo)
* Sentence Transformers
* FAISS
* Tavily API
* OpenRouter API

---

## 🚀 Future Improvements

* 🔹 Chat memory system
* 🔹 Conversation history retrieval
* 🔹 MongoDB Atlas (Cloud DB)
* 🔹 Authentication system
* 🔹 Frontend UI (React)

---

## 👩‍💻 Author

**Zainab Saeed**

---

## ⭐ Notes

This project is designed for learning and demonstration purposes. It follows a modular architecture to ensure scalability and maintainability.

---

💡 *Feel free to fork, improve, and share!* 🚀

