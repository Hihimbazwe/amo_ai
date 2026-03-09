# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AMO PAY AI Backend Server
# Pretrained Model: Groq (llama-3.3-70b-versatile)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import json
import os

# ── 1. Load API key from .env ──────────────────
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise Exception("❌ GROQ_API_KEY not found in .env file!")

client = Groq(api_key=api_key)
print("✅ Groq client connected successfully")

# ── 2. Load all your datasets ──────────────────
with open("data/app_knowledge.json", "r", encoding="utf-8") as f:
    app_knowledge = json.load(f)
    print("✅ app_knowledge.json loaded")

with open("data/faq.json", "r", encoding="utf-8") as f:
    faq = json.load(f)
    print("✅ faq.json loaded")

with open("data/kinya.json", "r", encoding="utf-8") as f:
    kinya = json.load(f)
    print("✅ kinya.json loaded")

# ── 3. Build the master system prompt ─────────
SYSTEM_PROMPT = f"""
You are AMO, the official AI assistant for Amo Pay — 
a Rwandan fintech app for money transfers, currency 
exchange, and merchant services.

You were built by the Amo Pay team to help users 
navigate and use the app confidently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APP KNOWLEDGE BASE:
{json.dumps(app_knowledge, indent=2, ensure_ascii=False)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FREQUENTLY ASKED QUESTIONS:
{json.dumps(faq, indent=2, ensure_ascii=False)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KINYARWANDA KNOWLEDGE:
{json.dumps(kinya, indent=2, ensure_ascii=False)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR RULES — FOLLOW THESE STRICTLY:

1. LANGUAGE DETECTION:
   - Detect the language the user writes in
   - If they write in Kinyarwanda → reply ONLY in Kinyarwanda
   - If they write in English → reply ONLY in English
   - If they mix both → reply in whichever language dominates
   - NEVER switch languages unless the user switches first

2. KNOWLEDGE BOUNDARIES:
   - Only answer based on the app knowledge provided above
   - If a question is not covered in your knowledge base,
     say honestly: "I don't have that information yet. 
     Please contact support@amopay.com"
   - NEVER make up features or information

3. THINGS YOU CAN HELP WITH:
   - How to send money
   - How to exchange currencies
   - How to become a merchant
   - How to complete KYC
   - Understanding fees and limits
   - App navigation guidance
   - Transaction status questions
   - Security tips

4. THINGS YOU CANNOT HELP WITH:
   - Reversing or cancelling transactions
   - Changing account passwords directly
   - Accessing user account details
   - Legal or tax advice
   - For these, always say: 
     "Please contact support@amopay.com or call +250 700 000 000"

6. TONE & STYLE:
   - Be warm, professional, and concise.
   - Use plain text for step-by-step instructions (e.g., 1., 2., 3.).
   - NEVER use markdown symbols like stars (**), hashtags (###), or underscores (__).
   - Keep responses short and clear.
   - Always be encouraging and helpful.

7. KINYARWANDA QUALITY:
   - Use natural, high-quality Kinyarwanda.
   - AVOID Swahili-isms (e.g., NEVER use "Wasiliana", "Nakufasha", or "Asante").
   - Use "Vugana", "Nagufasha", and "Murakoze" instead.
   - Follow the grammar and terminology provided in KINYARWANDA KNOWLEDGE.

8. ESCALATION:
   - For suspicious activity → tell user to contact support immediately.
   - For suspended accounts → direct to support@amopay.com.
   - For failed KYC multiple times → direct to support@amopay.com.
"""

# ── 4. Create the API server ───────────────────
app = FastAPI(
    title="AMO Pay AI Backend",
    description="AI assistant backend for Amo Pay fintech app",
    version="1.0.0"
)

# Allow React Native app to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 5. Define request format ───────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []

# ── 6. Main chat endpoint ──────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Start with the system prompt
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        # Add conversation history so AMO remembers context
        for msg in request.history:
            messages.append({
                "role": "user" if msg.get("sender") == "user" else "assistant",
                "content": msg.get("text", "")
            })

        # Add the new user message
        messages.append({
            "role": "user",
            "content": request.message
        })

        # Call Groq with Llama 3.3 70B
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # best free multilingual model
            messages=messages,
            max_tokens=600,
            temperature=0.7,       # balanced creativity vs accuracy
            top_p=0.9,
            stream=False,
        )

        reply = response.choices[0].message.content

        return {
            "reply": reply,
            "status": "success",
            "model": "llama-3.3-70b-versatile"
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"AMO AI error: {str(e)}"
        )

# ── 7. Health check endpoint ───────────────────
@app.get("/")
def health():
    return {
        "status": "✅ AMO Pay AI server is running",
        "model": "Groq — llama-3.3-70b-versatile",
        "languages": ["English", "Kinyarwanda"],
        "datasets_loaded": [
            "app_knowledge.json",
            "faq.json",
            "kinya.json"
        ]
    }

# ── 8. Test endpoint (check if datasets loaded) ─
@app.get("/test")
def test():
    return {
        "app_name": app_knowledge["app_identity"]["name"],
        "faq_count": len(faq["faqs"]),
        "languages": app_knowledge["local_context"]["languages"],
        "status": "All datasets loaded correctly ✅"
    }


    # Add this at the very bottom of server.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)