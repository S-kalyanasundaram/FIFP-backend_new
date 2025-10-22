#prediction.py
import re
import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------
# Load Environment & Initialize OpenAI
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not found in environment variables")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Helper Functions
# -----------------------------
def mask_sensitive(text: str) -> str:
    """Mask sensitive information like emails, phone numbers, and card numbers."""
    text = re.sub(r'([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', r'\1***\2', text)
    text = re.sub(r'(\+?\d[\d\-\s]{8,}\d)', lambda m: "****" + m.group(0)[-4:], text)
    text = re.sub(r'((?:\d[ -]?){13,19})', lambda m: "**** **** **** " + re.sub(r'\D', '', m.group(0))[-4:], text)
    return text


def flatten_doc(doc: Any) -> str:
    """Convert a dict or Document into plain text string."""
    if isinstance(doc, dict):
        return " ".join([f"{k}: {v}" for k, v in doc.items() if v])
    return getattr(doc, "page_content", str(doc))


def extract_relevant_data(all_docs: List[Any], question: str) -> str:
    """Extract only relevant portions of user data based on keywords in the question."""
    question_keywords = [word.lower() for word in re.findall(r'\w+', question)]
    relevant_texts = []

    for doc in all_docs:
        content = flatten_doc(doc)
        content_lower = content.lower()

        if any(kw in content_lower for kw in question_keywords):
            relevant_texts.append(content)

    if not relevant_texts:
        relevant_texts = [flatten_doc(doc) for doc in all_docs]

    return "\n\n".join(relevant_texts)


def format_prediction_output(question: str, result: Dict[str, Any]) -> str:
    """Format the prediction result into neat, user-friendly text."""
    if result.get("prediction") is None:
        return (
            f"❌ Could not predict for:\n➡️ {question}\n\n"
            f"Reason: {result.get('explanation', 'No explanation provided')}"
        )

    return (
        "💡 Answer to Your Question\n\n"
        f"➡️ {question}\n\n"
        f"📌 Prediction: {result['prediction']}\n"
        f"📝 Explanation: {result['explanation']}"
    )

# -----------------------------
# Prediction Function
# -----------------------------
def generate_prediction(user_id: str, question: str) -> str:
    from main import load_data  # Lazy import to avoid circular import

    try:
        all_docs = load_data(user_id)
        if not all_docs:
            return "No data available to generate prediction."

        context_text = extract_relevant_data(all_docs, question)

        system_prompt = (
            "You are a financial prediction assistant.\n"
            "You MUST use the provided user data to answer.\n"
            "Always return ONLY valid JSON, nothing else:\n"
            "{ \"prediction\": <number or null>, \"explanation\": \"short reasoning\" }"
        )

        user_prompt = f"User Question: {question}\n\nUser Data:\n{context_text}"

        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        raw_answer = resp.choices[0].message.content.strip()
        raw_answer = mask_sensitive(raw_answer)

        # ✅ Try to parse JSON safely
        try:
            result = json.loads(raw_answer)
        except json.JSONDecodeError:
            result = {"prediction": None, "explanation": "Could not parse model response."}

        # ✅ Return neatly formatted text
        return format_prediction_output(question, result)

    except Exception as e:
        import traceback
        return f"❌ Error generating prediction: {e}\n{traceback.format_exc()}"
