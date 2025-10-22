#utils.py
from typing import List
from pymongo import MongoClient
from bson import ObjectId
from langchain.schema import Document
import os
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
DB_NAME = "FIRE"

if not mongo_uri:
    raise RuntimeError("❌ MONGO_URI missing in .env")

client = MongoClient(mongo_uri)

COLLECTIONS = [
    "networths", "personalrisks", "net_worths", "multiusers", "mfdetails",
    "marriagefundplans", "insurances", "houseplans", "googles", "fundallocations",
    "childexpenses", "childeducations", "budgetincomeplans", "firequestions",
    "financials", "customplans", "expensesmasters", "vehicles", "emergencyfunds",
    "profiles", "realitybudgetincomes"
]


def load_user_documents(user_id: str) -> List[Document]:
    db = client[DB_NAME]
    all_docs = []
    for collection_name in COLLECTIONS:
        cursor = db[collection_name].find({"userId": user_id}, projection={"_id": 0, "userId": 0})
        for doc in cursor:
            text = f"[{collection_name}]\n" + "\n".join([f"{k}: {v}" for k, v in doc.items()])
            all_docs.append(Document(page_content=text))
        # ✅ Correct logging of documents
    return all_docs
