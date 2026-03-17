from pymongo import MongoClient
from config import MONGO_URI, DB_NAME
import datetime

client = MongoClient(MONGO_URI)

db = client[DB_NAME]

collection = db["chats"]

def save_chat(question, answer):

    data = {
        "user_input": question,
        "response": answer,
        "timestamp": str(datetime.datetime.now())
    }

    collection.insert_one(data)