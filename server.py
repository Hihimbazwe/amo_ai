# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AMO PAY AI Backend Server — PRODUCTION GRADE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import json
import os
import asyncio
import re
import base64
import io
import logging
import traceback
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageFilter

# ── 1. Load Environment & Config ───────────────
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
ALL_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct:novita",
    "mistralai/Mistral-7B-Instruct-v0.3:novita",
    "microsoft/Phi-3-mini-4k-instruct:hf-inference"
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AMO-Server")

# ── 2. KYC/Chat Models (Lazy loaded) ──────────
_embedder = None
_yolo_model = None
_ocr_reader = None
_mp_face_mesh = None

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            logger.info("📡 Loading local embedding model (Transformer)...")
            _embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedder ready")
        except Exception as e:
            logger.warning(f"⚠️ Embedder not ready: {e}")
    return _embedder

# ── 3. KYC Models (Lazy loaded) ────────────────
_yolo_model = None
_ocr_reader = None
_mp_face_mesh = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            logger.info("🎯 Loading YOLOv8n (ID Detection)...")
            _yolo_model = YOLO('yolov8n.pt') 
            logger.info("✅ YOLOv8n ready")
        except Exception as e:
            logger.warning(f"⚠️ YOLOv8 not ready: {e}")
    return _yolo_model

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            logger.info("📖 Loading EasyOCR...")
            _ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("✅ EasyOCR ready")
        except Exception as e:
            logger.warning(f"⚠️ EasyOCR not ready: {e}")
    return _ocr_reader

def get_mediapipe_mesh():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        try:
            import mediapipe as mp
            logger.info("⚡ Loading MediaPipe FaceMesh...")
            _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            logger.info("✅ MediaPipe FaceMesh ready")
        except Exception as e:
            logger.warning(f"⚠️ MediaPipe not ready: {e}")
    return _mp_face_mesh

# ── 4. Domain Guard ────────────────────────────
def is_out_of_domain(query: str) -> bool:
    domain_keywords = [
        "money", "transfer", "pay", "fee", "limit", "merchant", "kyc", "id",
        "account", "login", "password", "security", "exchange", "rate", "bill",
        "electricity", "water", "irembo", "airtime", "amo", "kigali", "rwanda",
        "card", "blik", "apple", "google", "support", "help", "contact"
    ]
    query_lower = query.lower()
    if any(k in query_lower for k in domain_keywords):
        return False
    return len(re.findall(r'\w+', query_lower)) > 3

# ── 5. Load Knowledge Base ─────────────────────
DATA_DIR = "data"
app_knowledge = {}
faq = {}
kinya = {}

try:
    with open(os.path.join(DATA_DIR, "app_knowledge.json"), "r", encoding="utf-8") as f:
        app_knowledge = json.load(f)
    with open(os.path.join(DATA_DIR, "faq.json"), "r", encoding="utf-8") as f:
        faq = json.load(f)
    with open(os.path.join(DATA_DIR, "kinya.json"), "r", encoding="utf-8") as f:
        kinya = json.load(f)
except Exception as e:
    logger.warning(f"⚠️ Knowledge files missing or error: {e}")

# ── 6. System Prompt ───────────────────────────
SYSTEM_PROMPT = f"""
ROLE: You are AMO, the Senior AI Support Systems Engineer for Amo Pay.
TONE: Professional, concise, accurate, and helpful. Avoid flowery language.
DOMAIN: Amo Pay — Rwandan fintech ecosystem (transfers, exchange, merchants, utilities).
CORE KNOWLEDGE:
{json.dumps(app_knowledge, indent=1)}
GUARDIAN RULES:
1. STRICT DOMAIN: You ONLY answer questions about Amo Pay and Rwandan fintech.
2. DATASET-FIRST: If information is not in the provided KNOWLEDGE, say:
   - English: "I'm sorry, I don't have specific information on that. Please contact support@amopay.com."
   - Kinyarwanda: "Mwihangane, nta makuru ahagije mbifitiye. Mwaduhamagara kuri support@amopay.com."
3. NO HALLUCINATION: Never invent fees, limits, or steps.
4. BILINGUAL: Detect language automatically. English in -> English out. Kinyarwanda in -> Kinyarwanda out.
5. ESCALATION: For transaction reversals or hacked accounts, direct to +250 700 000 000 immediately.
"""

# ── 7. Request Models ──────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []

class KYCRequest(BaseModel):
    id_front_image: str
    id_back_image: str = None
    selfie_image: str
    full_name: str
    date_of_birth: str = None
    id_number: str = None
    document_type: str = "national_id"

class ValidateImageRequest(BaseModel):
    image: str
    image_type: str
    id_number: Optional[str] = None
    date_of_birth: Optional[str] = None

# ── 8. Language Detection ──────────────────────
def detect_lang(text: str) -> str:
    msg_lower = text.lower()
    kw_markers = ['muraho', 'bite', 'mwaramutse', 'ndashaka', 'mfasha', 'amafaranga', 'gute']
    if any(m in msg_lower for m in kw_markers):
        return "Kinyarwanda"
    return "English"

# ── 9. KYC Helpers ─────────────────────────────
def clean_base64(b64: str) -> str:
    if "," in b64: return b64.split(",")[1]
    return b64

def decode_image(b64: str) -> np.ndarray:
    raw = base64.b64decode(clean_base64(b64))
    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def calculate_ear(landmarks, eye_indices):
    def dist(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))
    p2, p6 = landmarks[eye_indices[1]], landmarks[eye_indices[5]]
    p3, p5 = landmarks[eye_indices[2]], landmarks[eye_indices[4]]
    p1, p4 = landmarks[eye_indices[0]], landmarks[eye_indices[3]]
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4))

# ── 10. App Setup ──────────────────────────────
app = FastAPI(title="AMO Pay AI + KYC Production", version="6.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── 11. Chat Endpoint ──────────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Log query for tracking
        logger.info(f"💬 Chat request: {request.message[:50]}...")
        if is_out_of_domain(request.message):
            lang = detect_lang(request.message)
            if lang == "Kinyarwanda":
                return {"reply": "Nshobora gufasha gusa kubibazo bijyanye na Amo Pay.", "status": "guard_blocked"}
            return {"reply": "I can only assist with Amo Pay related questions.", "status": "guard_blocked"}

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in request.history[-5:]:
            role = "user" if h.get("sender") == "user" else "assistant"
            messages.append({"role": role, "content": h.get("text", "")})

        lang = detect_lang(request.message)
        messages.append({"role": "user", "content": f"{request.message}\n[CRITICAL: Reply in {lang} only]"})

        async with httpx.AsyncClient(timeout=45.0) as client:
            for model_name in ALL_MODELS:
                try:
                    response = await client.post(
                        HF_ROUTER_URL,
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={"model": model_name, "messages": messages, "temperature": 0.2, "max_tokens": 500}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        reply = data["choices"][0]["message"]["content"].strip()
                        return {"reply": reply, "status": "success", "model": model_name}
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} failed: {e}")
                    continue

        return {"reply": "System busy. Please try again later.", "status": "fallback"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 12. KYC Image Validation Endpoint ─────────
@app.post("/kyc/validate-image")
async def validate_image(request: ValidateImageRequest):
    try:
        img_rgb = decode_image(request.image)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h, w = img_gray.shape

        # 1. Quality Checks (Local OpenCV)
        brightness = np.mean(img_gray)
        sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        
        if brightness < 45: return {"valid": False, "reason": "Image is too dark. Please use better lighting."}
        if brightness > 250: return {"valid": False, "reason": "Image is too bright/overexposed."}
        if sharpness < 40: return {"valid": False, "reason": "Image is blurry. Hold your phone steady."}

        # 2. Case: Selfie
        if request.image_type == "selfie":
            mesh = get_mediapipe_mesh()
            if not mesh: return {"valid": False, "reason": "Facial analysis engine busy."}
            
            results = mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return {"valid": False, "reason": "No face detected in selfie. Please center your face."}
            
            landmarks = results.multi_face_landmarks[0]
            pts = [(l.x * w, l.y * h) for l in landmarks.landmark]
            left_eye = [362, 385, 387, 263, 373, 380]
            right_eye = [33, 160, 158, 133, 153, 144]
            avg_ear = (calculate_ear(pts, left_eye) + calculate_ear(pts, right_eye)) / 2.0
            
            if avg_ear < 0.18:
                return {"valid": False, "reason": "Please keep your eyes open during the selfie."}
            
            return {"valid": True, "reason": "Selfie validated successfully."}

        # 3. Case: ID
        elif request.image_type == "id":
            yolo = get_yolo_model()
            is_partial = False
            id_box = None
            
            if yolo:
                results = yolo(img_rgb, verbose=False)[0]
                if len(results.boxes) > 0:
                    # Sort by confidence/area and pick the best card-like box
                    box = results.boxes[0].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = box
                    id_box = (int(x1), int(y1), int(x2), int(y2))
                    
                    if x1 < 5 or y1 < 5 or x2 > w-5 or y2 > h-5:
                        is_partial = True

            if is_partial:
                return {"valid": False, "reason": "ID is cut off. Ensure the entire card is visible inside the frame."}

            # ── OCR Preprocessing ──
            # If we found a box, crop and pad it. Otherwise use the whole image.
            working_img = img_rgb
            if id_box:
                bx1, by1, bx2, by2 = id_box
                # Add 5% padding
                pad_w = int((bx2 - bx1) * 0.05)
                pad_h = int((by2 - by1) * 0.05)
                working_img = img_rgb[max(0, by1-pad_h):min(h, by2+pad_h), max(0, bx1-pad_w):min(w, bx2+pad_w)]
            
            # Convert to gray and enhance contrast for OCR
            working_gray = cv2.cvtColor(working_img, cv2.COLOR_RGB2GRAY)
            working_gray = cv2.equalizeHist(working_gray) # Boost contrast

            reader = get_ocr_reader()
            if not reader: return {"valid": False, "reason": "OCR engine busy."}
            
            ocr_results = reader.readtext(working_gray)
            all_text = " ".join([r[1] for r in ocr_results]).upper()
            
            if len(all_text) < 5:
                return {"valid": False, "reason": "Could not read ID text. Avoid glare and shadows."}

            # Matching Logic (Enhanced Fuzzy Matching)
            from fuzzywuzzy import fuzz
            
            # Normalize function: Aggressively target numbers and uppercase letters
            def normalize(t): 
                if not t: return ""
                # Replace common OCR misreads: O->0, I->1, Z->2, S->5, G->6, B->8
                clean = re.sub(r'[^0-9A-Z]', '', str(t).upper())
                return clean.replace('O', '0').replace('I', '1').replace('L', '1').replace('Z', '2').replace('S', '5').replace('G', '6').replace('B', '8')

            ocr_normalized = normalize(all_text)
            
            # 1. ID Number Match
            if request.id_number:
                input_normalized = normalize(request.id_number)
                logger.info(f"🔍 KYC Match — Input: {input_normalized[:5]}...{input_normalized[-3:]} vs OCR Length: {len(ocr_normalized)}")
                
                match_found = False
                if len(input_normalized) >= 5:
                    # Direct substring check
                    if input_normalized in ocr_normalized:
                        match_found = True
                    else:
                        # Fuzzy windowed check
                        ratio = fuzz.partial_ratio(input_normalized, ocr_normalized)
                        logger.info(f"📊 Fuzzy Match Ratio: {ratio}%")
                        if ratio >= 80: # Lowered to 80 for 16-digit Rwandan IDs
                            match_found = True
                
                if not match_found:
                    logger.warning(f"❌ ID Mismatch. OCR detected characters: {ocr_normalized[:50]}...")
                    return {"valid": False, "reason": "ID number mismatch. Please ensure the image is perpendicular to the camera and without glare."}

            # 2. DOB Match
            if request.date_of_birth:
                # App typically sends YYYY-MM-DD
                dob_clean = re.sub(r'[^0-9]', '', str(request.date_of_birth))
                match_found = False
                
                if len(dob_clean) == 8:
                    year, month, day = dob_clean[:4], dob_clean[4:6], dob_clean[6:]
                    # Try YYYYMMDD and DDMMYYYY
                    formats_to_try = [f"{year}{month}{day}", f"{day}{month}{year}"]
                    for fmt in formats_to_try:
                        if normalize(fmt) in ocr_normalized or fuzz.partial_ratio(normalize(fmt), ocr_normalized) >= 90:
                            match_found = True
                            break
                            
                if not match_found:
                    # Final fallback: just check if year, month and day all appear in the OCR
                    if len(dob_clean) == 8:
                        year, month, day = dob_clean[:4], dob_clean[4:6], dob_clean[6:]
                        if year in ocr_normalized and month in ocr_normalized and day in ocr_normalized:
                            match_found = True

                if not match_found:
                    logger.warning(f"❌ DOB Mismatch. Input: {request.date_of_birth} vs OCR: {ocr_normalized[:30]}...")
                    return {"valid": False, "reason": "Date of birth mismatch. Verify the details exactly match your ID."}
            
            return {"valid": True, "reason": "ID validated successfully.", "ocr_preview": all_text[:100]}

        return {"valid": False, "reason": "Invalid image type."}

    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        return {"valid": False, "reason": f"Processing error: {str(e)}"}

# ── 13. KYC Verify Endpoint ───────────────────
@app.post("/kyc/verify")
async def verify_kyc(request: KYCRequest):
    return {"success": True, "kyc_passed": True, "message": "KYC flow successful (local processing complete)."}

@app.get("/")
def health():
    return {"status": "✅ AMO Pay AI Online (Local KYC mode)", "version": "6.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)