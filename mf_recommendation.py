# mf_recommendation.py

import os
import re
import json
import pandas as pd
from pymongo import MongoClient
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from utils import load_user_documents

# -----------------------------
# Load Environment & OpenAI
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY missing in environment")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Helper Functions
# -----------------------------
def extract_user_financials(all_docs: List[Any]) -> Dict[str, float]:
    """Extract salary, savings, loan, retirement info from user docs."""
    user_data = {
        "monthly_salary": 0.0,
        "monthly_income": 0.0,
        "savings": 0.0,
        "loan_amount": 0.0,
        "retirement_plan": ""
    }

    for doc in all_docs:
        text = getattr(doc, "page_content", str(doc)).lower()
        if "salary" in text:
            match = re.search(r"salary[:\- ]?(\d+)", text)
            if match:
                user_data["monthly_salary"] = float(match.group(1))
        if "income" in text:
            match = re.search(r"income[:\- ]?(\d+)", text)
            if match:
                user_data["monthly_income"] = float(match.group(1))
        if "savings" in text:
            match = re.search(r"savings[:\- ]?(\d+)", text)
            if match:
                user_data["savings"] = float(match.group(1))
        if "loan" in text:
            match = re.search(r"loan[:\- ]?(\d+)", text)
            if match:
                user_data["loan_amount"] = float(match.group(1))
        if "retirement" in text:
            user_data["retirement_plan"] = "yes"

    return user_data



def load_fund_data_from_mongo() -> pd.DataFrame:
    """Load mutual fund dataset from MongoDB collection 'mfdetails' in DB 'FIRE'."""
    mongo_uri = os.getenv("MONGO_URI")
    db_name = "FIRE"  # ✅ fixed DB name

    if not mongo_uri:
        raise RuntimeError("❌ MONGO_URI missing in environment")

    client = MongoClient(mongo_uri)
    db = client[db_name]  # ✅ explicitly select FIRE DB
    collection = db["mfdetails"]

    # Fetch all documents
    data = list(collection.find({}, {"_id": 0}))

    if not data:
        raise RuntimeError("❌ No mutual fund data found in MongoDB collection 'mfdetails'")

    df = pd.DataFrame(data)

    # Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    return df

def recommend_funds(user_data: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Filter mutual funds based on user risk and financial situation."""

    # Step 1: Determine risk appetite
    if user_data["loan_amount"] > user_data["savings"]:
        risk_level = "low"
    elif user_data["monthly_salary"] > 100000:
        risk_level = "high"
    else:
        risk_level = "moderate"

    # Step 2: Filter by risk profile
    filtered = df[df["risk_profile"].str.lower() == risk_level]

    # Step 3: Rank funds (example: lower expense ratio, higher vr_rating)
    if not filtered.empty:
        filtered = filtered.sort_values(by=["vr_rating", "risk_rating"], ascending=[False, False])
        return filtered.head(5)  # top 5 recommendations

    return df.sort_values(by=["vr_rating"], ascending=False).head(5)


def explain_recommendations(user_data: Dict[str, Any], funds: pd.DataFrame, question: str) -> str:
    """Use OpenAI to explain why these funds are recommended."""
    context = f"User financials: {user_data}\n\nRecommended Funds:\n{funds.to_dict(orient='records')}"

    system_prompt = (
        "You are a financial advisor. Based on user financials and fund dataset, "
        "explain in simple terms why these funds are suitable. Keep it short and clear."
    )

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\n{context}"}
        ],
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()

def answer_general_question(question: str, df: pd.DataFrame) -> str:
    """Answer general mutual fund questions using OpenAI + fund dataset."""
    context = f"Available mutual fund data:\n{df.to_dict(orient='records')[:50]}"  # limit to 50 rows for context

    system_prompt = (
        "You are a financial assistant. Answer user questions about mutual funds. "
        "You may use the dataset provided, but if the answer is general (like 'what is a mutual fund'), "
        "give a simple clear explanation. "
        "If user asks to compare schemes, identify them from the dataset and provide comparison. "
        "If user asks for best/low risk/popular scheme, analyze dataset accordingly."
    )

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\n{context}"}
        ],
        temperature=0.3
    )
    final= f"💡 for you:\n\n{resp.choices[0].message.content.strip()}"

    return final


# -----------------------------
# Main function
# -----------------------------

def generate_fund_recommendation(user_id: str, question: str) -> str:
    """
    Generate mutual fund recommendations for a user based on their financial data.

    Args:
        user_id (str): The ID of the user.
        question (str): The question asked by the user.

    Returns:
        str: A formatted string with top fund recommendations and explanations.
    """
    try:
        # Load user documents
        all_docs = load_user_documents(user_id)
        if not all_docs:
            return "No user data available."

        # Extract user's financial data
        user_data = extract_user_financials(all_docs)

        # Load mutual fund data from MongoDB
        df = load_fund_data_from_mongo()

        # Get top recommended funds
        top_funds = recommend_funds(user_data, df)
        if top_funds.empty:
            return "No suitable mutual funds found."

        # Generate explanation for recommendations
        explanation = explain_recommendations(user_data, top_funds, question)

        # Prepare the formatted recommendation string
        return (
            "## 📊 Mutual Fund Recommendation\n\n"
            f"**Question Asked:** {question}\n\n"
            f"**Top Funds:**\n"
            f"{top_funds[['scheme_name', 'risk_profile', 'expense_ratio', 'vr_rating']].to_string(index=False)}\n\n"
            f"**Why these funds?**\n"
            f"{explanation}"
        )

    except Exception as e:
        import traceback
        return f"❌ Error in fund recommendation: {e}\n{traceback.format_exc()}"

