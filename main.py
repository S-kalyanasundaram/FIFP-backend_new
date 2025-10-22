# main.py
import re
import os
import numpy as np
import json
from pymongo import MongoClient
import ast
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv
from langchain.schema import Document
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
from prediction import generate_prediction
from utils import load_user_documents
from mf_recommendation import (
    load_fund_data_from_mongo,
    answer_general_question,
    generate_fund_recommendation,
)

# -----------------------------
# Config / ENV
# -----------------------------
df = load_fund_data_from_mongo()
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not mongo_uri or not OPENAI_API_KEY:
    raise RuntimeError("❌ MONGO_URI or OPENAI_API_KEY missing in .env")

# MongoDB Client
client = MongoClient(mongo_uri)
DB_NAME = "FIRE"
COLLECTIONS = [
    "networths", "personalrisks", "net_worths", "multiusers", "mfdetails",
    "marriagefundplans", "insurances", "houseplans", "googles", "fundallocations",
    "childexpenses", "childeducations", "budgetincomeplans", "firequestions",
    "financials", "customplans", "expensesmasters", "vehicles", "emergencyfunds",
    "profiles", "realitybudgetincomes"
]

# Chat History Collection
CHAT_HISTORY_COLLECTION = "chat_history"

# OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI App
app = FastAPI(title="Finance QA API", version="2.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for retrievers (per user)
retriever_cache: Dict[str, Dict] = {}

# -----------------------------
# Helper Functions
# -----------------------------
def validate_user_id(user_id: str):
    if not isinstance(user_id, str) or not user_id.strip():
        raise HTTPException(status_code=400, detail="Invalid user_id format.")


def cosine_sim(a: np.ndarray, b_matrix: np.ndarray):
    if b_matrix.size == 0:
        return np.array([])
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b_matrix / (np.linalg.norm(b_matrix, axis=1, keepdims=True) + 1e-10)
    return np.dot(b, a)


def embed_texts_openai(texts: List[str], model: str = "text-embedding-3-large"):
    if not texts:
        return np.array([])
    try:
        resp = openai_client.embeddings.create(model=model, input=texts)
        embeddings = [item.embedding for item in resp.data]
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI embedding error: {e}")


def mask_sensitive(text: str):
    text = re.sub(r'([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', r'\1***\2', text)
    text = re.sub(r'(\+?\d[\d\-\s]{8,}\d)', lambda m: "****" + m.group(0)[-4:], text)
    text = re.sub(r'((?:\d[ -]?){13,19})', lambda m: "**** **** **** " + re.sub(r'\D', '', m.group(0))[-4:], text)
    return text


def format_answer_with_currency(user_id: str, question: str, answer: str) -> str:
    currency = get_user_currency(user_id)
    currencies = [
        "$", "dollar", "Dollar", "DOLLAR", "usd", "USD",
        "€", "euro", "Euro", "EURO", "eur", "EUR",
        "£", "pound", "Pound", "POUND", "gbp", "GBP",
        "¥", "yen", "Yen", "YEN", "jpy", "JPY",
        "₹", "rupee", "Rupee", "RUPEE", "inr", "INR",
        "CHF", "franc", "Franc", "FRANC", "chf",
        "C$", "cad", "Cad", "CAD", "Canadian Dollar", "canadian dollar",
        "A$", "aud", "Aud", "AUD", "Australian Dollar", "australian dollar",
        "¥", "yuan", "Yuan", "CNY", "cny", "Renminbi", "renminbi",
        "₽", "ruble", "Ruble", "RUBLE", "RUB", "rub",
        "₩", "won", "Won", "WON", "KRW", "krw",
        "₺", "lira", "Lira", "LIRA", "TRY", "try",
        "AED", "SAR", "BDT", "PKR", "NZD", "SGD"
    ]
    sample=load_user_documents(user_id)
    sample2=preprocess_user_documents(sample)
    sample3=print_structured_data(sample2)
    for curen in currencies:
        answer = answer.replace(curen, currency)

    formatted = f"""
## 💡 Answer to Your Question  

**Question Asked:**
➡️ {question}  

**📌 Here’s what I found:**  

{answer}

---

✨ *Tip:* Stay consistent with your savings & review your plan regularly!
"""
    return formatted


# -----------------------------
# Load user documents
# -----------------------------
def load_user_documents(user_id: str) -> List[Document]:
    db = client[DB_NAME]
    all_docs = []
    for collection_name in COLLECTIONS:
        cursor = db[collection_name].find({"userId": user_id}, projection={"_id": 0, "userId": 0})
        for doc in cursor:
            text = f"[{collection_name}]\n" + "\n".join([f"{k}: {v}" for k, v in doc.items()])
            all_docs.append(Document(page_content=text))
    return all_docs

def preprocess_user_documents(documents: List[Document]) -> Dict[str, List[dict]]:
    """
    Preprocess a list of Documents and return structured data grouped by collection name.
    """
    structured_data = {}

    for doc in documents:
        content = doc.page_content
        # Extract collection name from [collection_name] header
        match = re.match(r"\[(.*?)\]", content)
        if not match:
            continue
        collection_name = match.group(1)

        # Remove header line
        content_body = "\n".join(content.split("\n")[1:])

        # Convert the content into a dict safely
        collection_items = {}
        lines = content_body.split("\n")
        for line in lines:
            if not line.strip():
                continue
            key_value = line.split(":", 1)
            if len(key_value) != 2:
                continue
            if line =="createdAt:":
                    continue
            key = key_value[0].strip()
            value = key_value[1].strip()
            # Try to convert string representations of lists/dicts/numbers
            try:
                value = ast.literal_eval(value)
            except:
                pass
            collection_items[key] = value

        # Append to the structured data
        if collection_name not in structured_data:
            structured_data[collection_name] = []
        structured_data[collection_name].append(collection_items)

    return structured_data


def print_structured_data(structured_data: Dict[str, List[dict]]):
    output_lines = []
    for collection, items in structured_data.items():
        output_lines.append(f"**{collection}:**")
        if len(items) == 1:
            # Single document → print attributes: values
            for key, value in items[0].items():
                if key in ["createdAt", "updatedAt", "__v",'_id']:
                    continue
                keywords=["RetirementCalculations:","withdrawPlan:","investmentAchievementPlan:"]
                for i in keywords:
                    if key in i:
                        output_lines.append(f"**{key}**: {value}")
                else:
                    output_lines.append(f"{key}: {value}")
        else:
            output_lines.append("    [")
            for doc in items:
                # Remove unwanted keys before printing
                filtered_doc = {
                    k: v for k, v in doc.items()
                    if k not in ["createdAt", "updatedAt", '_id',"time","__v"]
                }
                doc_str = json.dumps(filtered_doc, ensure_ascii=False, indent=4)
                output_lines.append(f"   anna {doc_str},")
            output_lines.append("    ]")
        output_lines.append("")
        output_lines.append("-------------------------------------------------------------------")
    return "\n".join(output_lines)


def build_local_retriever(docs: List[Document], model: str = "text-embedding-3-small"):
    if not docs:
        return {"docs": [], "embeddings": np.array([]), "norms": np.array([]), "metadata": []}

    texts = [d.page_content for d in docs]
    metadata = [{"length": len(d.page_content)} for d in docs]
    vectors = embed_texts_openai(texts, model=model)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    vectors_normalized = vectors / norms
    return {
        "docs": docs,
        "embeddings": vectors_normalized,
        "norms": norms.flatten(),
        "metadata": metadata
    }


def get_top_k_docs(retriever_obj: Dict, question: str, k: int = 1, sim_threshold: float = 0.1):
    doc_embeddings = retriever_obj.get("embeddings", np.array([]))
    docs = retriever_obj.get("docs", [])
    if doc_embeddings.size == 0 or not docs:
        return []

    qvecs = embed_texts_openai([question], model="text-embedding-3-small")
    if qvecs.size == 0:
        return []

    sims = cosine_sim(qvecs[0], doc_embeddings)
    top_idx = np.argsort(-sims)[:k]
    top_scores = sims[top_idx]
    top_docs = [docs[idx] for idx, score in zip(top_idx, top_scores) if score >= sim_threshold]
    return top_docs


def get_user_currency(user_id: str) -> str:
    db = client[DB_NAME]
    profile = db["profiles"].find_one({"userId": user_id}, {"currency": 1})
    if profile and "currency" in profile:
        return profile["currency"]
    return "₹"


def Find_Finance_questions(question):
    classifier_prompt = f"""
    Return 0 ,You are an AI that gives only factual and information-based answers.only give text
    Return 0 ,if Greetings related questions
    Return 0 , if Profile or personal related question
    Return 1,If the user asks for code and non relavent questions instead of based information
    Question: "{question}"
    Return 0,if finance-related or, 1 if not.
    Return 1,if without finance related questions.
    Return 1 if finance related words
    Only return 0 or 1.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": classifier_prompt}],
        temperature=0
    )
    result = response.choices[0].message.content.strip()
    return result

def prediction(question):
    result=Find_Finance_questions(question)
    if result=="0":
        classifier_prompt = f"""
        Return only one number based on the message content:
        Return 0 ,if given questions prediction related
        Retun 0 , if given questions feuture related questions
        Question: "{question}"
        return 1, if not prediction related questions
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": classifier_prompt}],
            temperature=0
            )
        result = response.choices[0].message.content.strip()
    return result

def mutual_fund_general(question):
    result=Find_Finance_questions(question)
    if result=="0":
        classifier_prompt = f"""
        Return only one number based on the message content:
        Return 0 ,if given questions mutual Fund Related
        Return 0, if the message is a GENERAL mutual fund question (e.g., definitions, benefits, performance, comparisons, market trends, how they work, SIP meaning).
        Retun 0 , if given questions Mutual Funds and general Mutual Fund related questions, like what is mutual fund , tell about specific name  mutual fund
        Question: "{question}"
        Return 1, if questions not mutual Fund Related 
        Return 1, if questions not general mutual Fund Related 
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": classifier_prompt}],
            temperature=0
            )
        result = response.choices[0].message.content.strip()
    return result

def mutual_fund_recom(question):
    result=Find_Finance_questions(question)
    if result=="0":
        classifier_prompt = f"""
        Return only one number based on the message content:
        Return 0 ,if given questions mutual Fund recommend Related
        Retun 0 , if given questions Mutual Funds with user specific dataset related questions like best mutual fund for me
        Return 0 → if the message is a USER-SPECIFIC mutual fund question (e.g., best fund for me, based on my income, goals, duration, SIP plan, portfolio, or personal situation).Be smart enough to detect user-specific questions even if the word “me” or “my” isn’t directly mentioned but is implied (e.g., “best mutual fund for 3 years” → user-specific).
        Question: "{question}"
        Return 1, if questions not mutual Fund recommend Related 
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": classifier_prompt}],
            temperature=0
            )
        result = response.choices[0].message.content.strip()
    return result
        
        
    
#def generate_answer(user_id,question: str, top_context_docs: List[Document], all_docs: List[Document]):
def generate_answer(question: str,sample3):
    #context_text = "\n\n".join(d.page_content for d in top_context_docs)
    
    if not sample3.strip():
        return "I couldn’t find this information in your records."

    system_prompt = (
        "You are a financial assistant. "
        "Answer ONLY using the provided user database context."
        "Do not generate generic advice. "
        "If the context does not have enough information, reply with: "
        "I couldn’t find this information in your records.'"
    )
    
    result=Find_Finance_questions(question)
    if result == "0":
        prompt = f"""
        Question: "{question}"
        i have attached entire my financial data.
        i have attached my money management dataset.
        Use a table **only if the user’s question clearly requires structured, comparative, or tabular data** (such as comparisons, lists of items with multiple attributes, or organized data).
        Do not use a table unless it makes the answer clearer and more organized.
        Always keep tables simple and well-formatted with headers.
        To check whether the user's query is consistent with the given data.
        Check if the given data is available for the question the user is asking.
        if have data from given data, retrieve answer from this data.if not return related msg like data is not available that is Finance related.
        every question refrence the dataset for better understand to user.
        if user make felling mistake correct it and give answer
        must be refer the data set , and derive answer from the dataset , give answer based on the dataset.
        do not over write , if not amount or numbers i n the dataset
        do not use example , without dataset numbers.
        if user use words"me", shoult be answer is based on the given dataset 
    """
        user_prompt = f"User Question: {prompt}\n\nUser Data Context:\n{sample3}"
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content":"You are a helpful assistant"},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            raw_answer = resp.choices[0].message.content.strip()
            return raw_answer
        except Exception as e:
            return f"❌ Error in OpenAI API: {e}"
    else:
        return "I couldn’t find this information in your records."


# -----------------------------
# Chat History
# -----------------------------
def save_chat_history(user_id: str, question: str, answer: str, session_id: Optional[str] = None):
    db = client[DB_NAME]
    if not session_id:
        session_id = str(ObjectId())
    chat_entry = {
        "userId": user_id,
        "sessionId": session_id,
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow()
    }
    db[CHAT_HISTORY_COLLECTION].insert_one(chat_entry)
    return session_id


def get_user_name(user_id: str) -> Optional[str]:
    db = client[DB_NAME]
    profile = db["profiles"].find_one(
        {"userId": user_id},
        {"firstName": 1, "lastName": 1, "_id": 0}
    )
    if not profile:
        return None
    first = profile.get("firstName", "")
    last = profile.get("lastName", "")
    return f"{first} {last}".strip()


# -----------------------------
# API Models
# -----------------------------
class QuestionRequest(BaseModel):
    user_id: str
    question: str
    session_id: Optional[str] = None


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Finance QA API is running"}


@app.post("/load-data/{user_id}")
def load_data(user_id: str):
    validate_user_id(user_id)
    docs = load_user_documents(user_id)
    if not docs:
        raise HTTPException(status_code=404, detail="No documents found for this user")
    retriever_cache[user_id] = build_local_retriever(docs)
    return docs


@app.get("/user-name/{user_id}")
def fetch_user_name(user_id: str):
    validate_user_id(user_id)
    user_name = get_user_name(user_id)
    if not user_name:
        raise HTTPException(status_code=404, detail="User name not found")
    return user_name


@app.post("/ask")
def ask_question(req: QuestionRequest):
    user_id = req.user_id.strip()
    question = req.question.strip()
    session_id = req.session_id

    greetings = ["hi", "hello", "hey", "hii", "hiii", "hola"]
    greetings2 = ["ok", "bye", "bya", "goodbay", "thankyou", "tq", "good"]

    if any(re.search(rf"\b{g}\b", question.lower()) for g in greetings):
        answer = f"{question} 👋! I can help you only with your finance-related queries."
        session_id = save_chat_history(user_id, question, answer, session_id)
        return {"answer": answer, "session_id": session_id}

    if any(g in question.lower() for g in greetings2):
        answer = f"{question} Great! Let's keep your money working for you. 💹"
        session_id = save_chat_history(user_id, question, answer, session_id)
        return {"answer": answer, "session_id": session_id}
    
    mf_keywords = mutual_fund_recom(question)
    if mf_keywords=="0":
        answer = generate_fund_recommendation(user_id, question)
        formatted_answer = format_answer_with_currency(user_id, question, answer)
        session_id = save_chat_history(user_id, question, answer, session_id)
        return {"answer": answer, "session_id": session_id}
    MUTUAL_FUND_KEYWORDS = mutual_fund_general(question)
    if MUTUAL_FUND_KEYWORDS=="0":
        answer = answer_general_question(question, df)
        formatted_answer = format_answer_with_currency(user_id, question, answer)
        session_id = save_chat_history(user_id, question, answer, session_id)
        return {"answer": answer, "session_id": session_id}

    prediction_keywords = prediction(question)
    if prediction_keywords=="0":
        answer = generate_prediction(user_id, question)
        formatted_answer = format_answer_with_currency(user_id, question, answer)
        session_id = save_chat_history(user_id, question, formatted_answer, session_id)
        return {"answer": formatted_answer, "session_id": session_id}

    if user_id not in retriever_cache:
        docs = load_user_documents(user_id)
        if not docs:
            raise HTTPException(status_code=404, detail="No documents found for this user")
        retriever_cache[user_id] = build_local_retriever(docs)
        
    sample=load_user_documents(user_id)
    sample2=preprocess_user_documents(sample)
    sample3=print_structured_data(sample2)
    retriever_obj = retriever_cache[user_id]
    all_docs = retriever_obj.get("docs", [])
    top_docs = get_top_k_docs(retriever_obj, question)
    answer = generate_answer(question,sample3)
    formatted_answer = format_answer_with_currency(user_id, question, answer)
    session_id = save_chat_history(user_id, question, formatted_answer, session_id)

    return { "answer": formatted_answer,"session_id": session_id}


@app.get("/chat-sessions/{user_id}")
def get_sessions(user_id: str):
    db = client[DB_NAME]
    sessions = db[CHAT_HISTORY_COLLECTION].aggregate([
        {"$match": {"userId": user_id}},
        {"$group": {
            "_id": "$sessionId",
            "latest": {"$max": "$timestamp"},
            "title": {"$first": "$question"}
        }},
        {"$sort": {"latest": -1}}
    ])
    result = []
    for s in sessions:
        title = s.get("title") or "Untitled"
        result.append({
            "session_id": s["_id"],
            "title": title[:50] + ("..." if len(title) > 50 else ""),
            "latest": s["latest"].isoformat() if s.get("latest") else None
        })
    return {"sessions": result}


@app.get("/chat-history/{user_id}/{session_id}")
def chat_history(user_id: str, session_id: str):
    db = client[DB_NAME]
    chats = (
        db[CHAT_HISTORY_COLLECTION]
        .find({"userId": user_id, "sessionId": session_id})
        .sort("timestamp", 1)
    )
    history = []
    for c in chats:
        c["_id"] = str(c["_id"])
        c["timestamp"] = c["timestamp"].isoformat() if c.get("timestamp") else None
        history.append(c)
    return {"history": history}


@app.delete("/chat-sessions/{user_id}/{session_id}")
def delete_session(user_id: str, session_id: str):
    db = client[DB_NAME]
    result = db[CHAT_HISTORY_COLLECTION].delete_many(
        {"userId": user_id, "sessionId": session_id}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "message": "Session deleted"}


@app.put("/chat-sessions/{user_id}/{session_id}")
def rename_session(user_id: str, session_id: str, data: dict = Body(...)):
    new_title = data.get("title")
    if not new_title:
        raise HTTPException(status_code=400, detail="Title is required")

    db = client[DB_NAME]
    first_msg = db[CHAT_HISTORY_COLLECTION].find_one(
        {"userId": user_id, "sessionId": session_id},
        sort=[("timestamp", 1)]
    )
    if not first_msg:
        raise HTTPException(status_code=404, detail="Session not found")

    db[CHAT_HISTORY_COLLECTION].update_one(
        {"_id": first_msg["_id"]},
        {"$set": {"question": new_title}}
    )
    return {"status": "success", "message": "Session renamed"}
