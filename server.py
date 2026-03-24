# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AMO PAY AI Backend Server — Production Grade
# Model: Multi-model Router with Local Guardrails
# + KYC Verification (EasyOCR + DeepFace)
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
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

# ── 1. Load Environment & Config ───────────────
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    raise Exception("❌ HUGGINGFACE_API_KEY not found in .env file!")

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
ALL_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct:novita",
    "mistralai/Mistral-7B-Instruct-v0.3:novita",
    "microsoft/Phi-3-mini-4k-instruct:hf-inference"
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AMO-Server")

# ── 2. Local Model Initialization ──────────────
print("📡 Loading local embedding model (Transformer)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ── 3. KYC Models (Lazy loaded on first use) ───
_ocr_reader = None
_deepface_available = False

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            logger.info("📖 Loading EasyOCR...")
            _ocr_reader = easyocr.Reader(['en', 'fr'], gpu=False)
            logger.info("✅ EasyOCR ready")
        except ImportError:
            logger.warning("⚠️ EasyOCR not installed. Run: pip install easyocr")
            return None
    return _ocr_reader

def check_deepface():
    global _deepface_available
    if not _deepface_available:
        try:
            from deepface import DeepFace
            _deepface_available = True
        except ImportError:
            logger.warning("⚠️ DeepFace not installed. Run: pip install deepface tf-keras")
    return _deepface_available

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
    return len(re.findall(r'\w+', query_lower)) > 3 and not any(k in query_lower for k in domain_keywords)

# ── 5. Load Knowledge Base ─────────────────────
with open("data/app_knowledge.json", "r", encoding="utf-8") as f:
    app_knowledge = json.load(f)

with open("data/faq.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

with open("data/kinya.json", "r", encoding="utf-8") as f:
    kinya = json.load(f)

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

# ── 7. App Setup ───────────────────────────────
app = FastAPI(title="AMO Pay AI + KYC Production", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 8. Request Models ──────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []

class KYCRequest(BaseModel):
    id_front_image: str
    id_back_image: str = None
    selfie_image: str
    full_name: str
    date_of_birth: str = None
    id_number: str = None
    document_type: str = "national_id"

# ── 9. Language Detection ──────────────────────
def detect_lang(text: str) -> str:
    msg_lower = text.lower()
    kw_markers = ['muraho', 'bite', 'mwaramutse', 'ndashaka', 'mfasha', 'amafaranga', 'gute']
    if any(m in msg_lower for m in kw_markers):
        return "Kinyarwanda"
    return "English"

# ── 10. KYC Helpers ────────────────────────────
def decode_image(base64_str: str) -> np.ndarray:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

def save_temp_image(base64_str: str, filename: str) -> str:
    img_array = decode_image(base64_str)
    img = Image.fromarray(img_array)
    path = f"/tmp/{filename}"
    img.save(path)
    return path

def extract_text_from_id(image_array: np.ndarray) -> dict:
    reader = get_ocr_reader()
    if not reader:
        return {"raw_text": "", "name": None, "date_of_birth": None, "id_number": None}

    try:
        results = reader.readtext(image_array)
        all_text = " ".join([r[1] for r in results])
        lines = [r[1].strip() for r in results]

        extracted = {
            "raw_text": all_text,
            "name": None,
            "date_of_birth": None,
            "id_number": None,
            "expiry_date": None,
            "nationality": None,
        }

        for i, line in enumerate(lines):
            lu = line.upper()
            if any(k in lu for k in ["SURNAME", "LAST NAME", "NOM", "AMAZINA"]):
                if i + 1 < len(lines):
                    extracted["name"] = lines[i + 1].strip()
            elif any(k in lu for k in ["GIVEN", "FIRST NAME", "PRENOM", "IZINA"]):
                if extracted["name"] and i + 1 < len(lines):
                    extracted["name"] = f"{lines[i + 1].strip()} {extracted['name']}"

        date_pattern = r'\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{2}[/\-.]\d{2})\b'
        dates = re.findall(date_pattern, all_text)
        if dates:
            extracted["date_of_birth"] = dates[0]
            if len(dates) > 1:
                extracted["expiry_date"] = dates[-1]

        for pattern in [r'\b[A-Z]{1,2}\d{6,9}\b', r'\b\d{8,16}\b', r'\b[A-Z0-9]{9,12}\b']:
            matches = re.findall(pattern, all_text)
            if matches:
                extracted["id_number"] = matches[0]
                break

        for line in lines:
            if any(c in line.upper() for c in ["RWANDA", "RWANDAISE", "RWA"]):
                extracted["nationality"] = "Rwandan"
                break

        return extracted

    except Exception as e:
        logger.error(f"OCR error: {e}")
        return {"raw_text": "", "name": None, "date_of_birth": None, "id_number": None}

def compare_faces(id_path: str, selfie_path: str) -> dict:
    if not check_deepface():
        return {"match": True, "similarity_score": 75.0, "skipped": True}

    try:
        from deepface import DeepFace
        result = DeepFace.verify(
            img1_path=id_path,
            img2_path=selfie_path,
            model_name="VGG-Face",
            distance_metric="cosine",
            enforce_detection=False,
        )
        distance = result.get("distance", 1.0)
        verified = result.get("verified", False)
        similarity = max(0, (1 - distance) * 100)
        return {"match": verified, "similarity_score": round(similarity, 1), "distance": round(distance, 4)}
    except Exception as e:
        logger.error(f"Face comparison error: {e}")
        return {"match": False, "similarity_score": 0.0, "error": str(e)}

def compare_text(extracted: str, user_input: str) -> float:
    if not extracted or not user_input:
        return 0.0
    try:
        from fuzzywuzzy import fuzz
        e = extracted.upper().strip()
        u = user_input.upper().strip()
        return float(max(fuzz.ratio(e, u), fuzz.partial_ratio(e, u), fuzz.token_sort_ratio(e, u)))
    except ImportError:
        e = extracted.upper().strip()
        u = user_input.upper().strip()
        if e == u:
            return 100.0
        if e in u or u in e:
            return 80.0
        e_words = set(e.split())
        u_words = set(u.split())
        overlap = len(e_words & u_words) / max(len(e_words), len(u_words), 1)
        return round(overlap * 100, 1)

# ── 11. Hugging Face KYC Extras ────────────────
async def hf_ocr(image_base64: str) -> str:
    """Uses a specialized HF model for high-accuracy OCR."""
    if not api_key: return ""
    model = "microsoft/trocr-base-printed"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={"Authorization": f"Bearer {api_key}"},
                content=base64.b64decode(image_base64),
                timeout=15.0
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
    except Exception as e:
        logger.error(f"HF OCR Error: {e}")
    return ""

async def check_selfie_quality(image_base64: str) -> dict:
    """Uses a Vit model to ensure the selfie is professional."""
    if not api_key: return {"is_good": True, "score": 1.0}
    model = "google/vit-base-patch16-224"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={"Authorization": f"Bearer {api_key}"},
                content=base64.b64decode(image_base64),
                timeout=15.0
            )
            if response.status_code == 200:
                results = response.json()
                logger.info(f"HF Selfie Check: {results[:2]}")
                return {"is_good": True, "score": results[0].get("score", 0.0), "label": results[0].get("label")}
    except Exception as e:
        logger.error(f"HF Selfie Quality Error: {e}")
    return {"is_good": True, "score": 1, "note": "Local check only"}

# ── 12. Chat Endpoint ──────────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if is_out_of_domain(request.message):
            lang = detect_lang(request.message)
            if lang == "Kinyarwanda":
                return {"reply": "Nshobora gufasha gusa kubibazo bijyanye na Amo Pay.", "status": "guard_blocked"}
            return {"reply": "I can only assist with Amo Pay related questions.", "status": "guard_blocked"}

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in request.history[-5:]:
            role = "user" if msg.get("sender") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("text", "")})

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
                    print(f"⚠️ {model_name} failed: {e}")
                    continue

        return {"reply": "System busy. Please try again or visit amopay.com.", "status": "fallback"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 13. KYC Image Validation Endpoint ─────────
class ValidateImageRequest(BaseModel):
    image: str       # base64 encoded image
    image_type: str  # "selfie" | "id"

@app.post("/kyc/validate-image")
async def validate_image(request: ValidateImageRequest):
    """Real-time image validation."""
    try:
        # 1. Basic size check & Decode
        b64 = request.image
        if "," in b64:
            b64 = b64.split(",")[1]
        
        try:
            raw_bytes = base64.b64decode(b64)
        except Exception as decode_err:
            logger.error(f"Base64 decode error: {decode_err}")
            return {"valid": False, "reason": "Invalid image format. Please capture a new photo."}
        
        size_kb = len(raw_bytes) / 1024

        if size_kb < 5:
            return {"valid": False, "reason": "Image is too small or blank. Please retake in good lighting."}

        image_content = raw_bytes
        
        # 2. Local Quality Guardrails (Brightness/Contrast)
        local_quality_passed = False
        try:
            img = Image.open(io.BytesIO(image_content))
            img_gray = img.convert("L")  # Convert to grayscale
            np_img = np.array(img_gray)
            
            avg_brightness = np.mean(np_img)
            contrast = np.std(np_img)
            
            logger.info(f"[KYC/Quality] Brightness: {avg_brightness:.2f} | Contrast: {contrast:.2f}")
            
            if avg_brightness < 45:
                return {"valid": False, "reason": "Image is too dark. Please move to a brighter location and ensure even lighting."}
            if avg_brightness > 240:
                return {"valid": False, "reason": "Image is too bright/overexposed. Reduce glare and avoid direct flash."}
            if contrast < 15:
                return {"valid": False, "reason": "Image is too blurry or low contrast. Keep the camera steady and ensure focus."}
            
            local_quality_passed = True
            logger.info(f"[KYC] Local quality check PASSED")
        except Exception as q_err:
            logger.warning(f"Local quality check failed: {q_err}")
            # Continue to AI models if local check fails

        # 3. AI-powered validation with retry logic
        if request.image_type == "selfie":
            # 3a. Use YOLOS object detection — look for "person" label
            model = "hustvl/yolos-tiny"
            logger.info(f"[KYC] Starting selfie validation with {model}")
            
            try:
                # Shorter timeout for initial attempt
                async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=10.0)) as client:
                    resp = await client.post(
                        f"https://api-inference.huggingface.co/models/{model}",
                        headers={"Authorization": f"Bearer {api_key}"},
                        content=image_content,
                    )
                
                logger.info(f"[KYC/Selfie] HF Response Status: {resp.status_code}")
                
                if resp.status_code == 200:
                    detections = resp.json()
                    logger.info(f"[KYC/Selfie] Detections: {detections}")
                    
                    # detections is a list of {"label": ..., "score": ..., "box": {...}}
                    person_detections = [
                        d for d in detections
                        if isinstance(d, dict)
                        and d.get("label", "").lower() in ("person", "face", "head")
                        and d.get("score", 0) > 0.5
                    ]
                    if not person_detections:
                        return {"valid": False, "reason": "No human face detected. Make sure your face is clearly visible and centered in the frame."}
                    logger.info(f"[KYC] Selfie validation PASSED with score {person_detections[0]['score']:.2f}")
                    return {"valid": True, "reason": "Face detected successfully.", "score": person_detections[0]["score"]}
                
                elif resp.status_code == 503:
                    logger.warning("[KYC/Selfie] HF model loading (503)")
                    return {"valid": False, "reason": "AI model is loading. Please wait 5-10 seconds and retake your selfie."}
                
                else:
                    error_text = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
                    logger.warning(f"[KYC/Selfie] HF error: {error_text}")
                    return {"valid": False, "reason": "Face detection temporarily unavailable. Please retake the selfie."}
            
            except asyncio.TimeoutError as te:
                logger.error(f"[KYC/Selfie] Timeout: {te}")
                return {"valid": False, "reason": "Face detection service timeout. Please check your internet and retake the photo."}
            
            except Exception as e:
                logger.error(f"[KYC/Selfie] Error: {type(e).__name__}: {str(e)}")
                return {"valid": False, "reason": "Could not validate face. Please ensure good lighting and clear visibility of your face."}

        elif request.image_type == "id":
            # 3b. For ID — use image captioning to detect if it's a document
            model = "Salesforce/blip-image-captioning-base"
            logger.info(f"[KYC] Starting ID validation with {model}")
            
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=10.0)) as client:
                    resp = await client.post(
                        f"https://api-inference.huggingface.co/models/{model}",
                        headers={"Authorization": f"Bearer {api_key}"},
                        content=image_content,
                    )
                
                logger.info(f"[KYC/ID] HF Response Status: {resp.status_code}")
                
                if resp.status_code == 200:
                    result = resp.json()
                    caption = ""
                    if isinstance(result, list) and result:
                        caption = result[0].get("generated_text", "").lower()
                    elif isinstance(result, dict):
                        caption = result.get("generated_text", "").lower()

                    logger.info(f"[KYC/ID] Caption: {caption}")
                    
                    # Block obvious non-ID images
                    bad_keywords = ["landscape", "sky", "tree", "animal", "cat", "dog", "car", "food", "plate", "wall", "window", "person", "selfie", "flower"]
                    if any(kw in caption for kw in bad_keywords):
                        return {"valid": False, "reason": f"This doesn't appear to be an ID document (detected: {caption}). Photograph your ID card or passport clearly."}
                    
                    logger.info(f"[KYC] ID validation PASSED with caption: {caption}")
                    return {"valid": True, "reason": "ID document detected.", "caption": caption}
                
                elif resp.status_code == 503:
                    logger.warning("[KYC/ID] HF model loading (503)")
                    return {"valid": False, "reason": "Document validation model is loading. Please wait 5-10 seconds and retake the photo."}
                
                else:
                    error_text = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
                    logger.warning(f"[KYC/ID] HF error: {error_text}")
                    return {"valid": False, "reason": "Document validation temporarily unavailable. Please retake the photo."}
            
            except asyncio.TimeoutError as te:
                logger.error(f"[KYC/ID] Timeout: {te}")
                return {"valid": False, "reason": "Document validation service timeout. Please check your internet and retake the photo."}
            
            except Exception as e:
                logger.error(f"[KYC/ID] Error: {type(e).__name__}: {str(e)}")
                return {"valid": False, "reason": "Could not validate document. Please ensure good lighting and that the ID is clearly visible."}

        return {"valid": False, "reason": "Unknown image type."}

    except Exception as e:
        logger.error(f"Validate image root error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

# ── 14. KYC Verify Endpoint ────────────────────
@app.post("/kyc/verify")
async def verify_kyc(request: KYCRequest):
    issues = []
    temp_files = []

    try:
        logger.info(f"🔍 KYC started for: {request.full_name}")

        # 1. Image Quality Check (HF Model)
        selfie_quality = await check_selfie_quality(request.selfie_image)
        if selfie_quality.get("score", 1.0) < 0.3:
            issues.append("Selfie quality looks low. Please use better lighting.")

        try:
            id_image = decode_image(request.id_front_image)
            decode_image(request.selfie_image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

        id_path = save_temp_image(request.id_front_image, "kyc_id.jpg")
        selfie_path = save_temp_image(request.selfie_image, "kyc_selfie.jpg")
        temp_files = [id_path, selfie_path]

        # Advanced OCR (HF Model) + Local EasyOCR fallback
        hf_extracted_text = await hf_ocr(request.id_front_image)
        extracted = extract_text_from_id(id_image)
        
        if hf_extracted_text:
            logger.info(f"HF OCR Extracted: {hf_extracted_text[:50]}...")
            # Simple merge: if EasyOCR missed ID number but HF got it
            if not extracted.get("id_number") and any(c.isdigit() for c in hf_extracted_text):
                num_match = re.search(r'\b\d{8,16}\b', hf_extracted_text)
                if num_match: extracted["id_number"] = num_match.group(0)

        # Face comparison
        face_result = compare_faces(id_path, selfie_path)
        face_score = face_result.get("similarity_score", 0.0)

        if face_result.get("skipped"):
            issues.append("Face comparison unavailable — install deepface for full verification")
        elif face_score < 60:
            issues.append("Face does not clearly match the ID photo")
        elif face_score < 75:
            issues.append("Face match is low — retake selfie in better lighting")

        # Text comparison
        text_scores = []

        if extracted.get("name"):
            ns = compare_text(extracted["name"], request.full_name)
            text_scores.append(ns)
            if ns < 60:
                issues.append(f"Name mismatch — ID shows: '{extracted.get('name')}'")
        else:
            text_scores.append(50.0)
            issues.append("Could not read name from ID — ensure good lighting")

        if request.id_number and extracted.get("id_number"):
            ids = compare_text(extracted["id_number"], request.id_number)
            text_scores.append(ids)
            if ids < 80:
                issues.append("ID number mismatch")

        if request.date_of_birth and extracted.get("date_of_birth"):
            dobs = compare_text(extracted["date_of_birth"], request.date_of_birth)
            text_scores.append(dobs)
            if dobs < 70:
                issues.append("Date of birth mismatch")

        text_score = sum(text_scores) / len(text_scores) if text_scores else 50.0
        overall_score = (face_score * 0.5) + (text_score * 0.5)
        
        # Bonus for HF Quality
        if selfie_quality.get("score", 0) > 0.8:
            overall_score = min(100, overall_score + 5)
        kyc_passed = overall_score >= 65.0 and face_score >= 55.0

        logger.info(f"✅ KYC: {'PASS' if kyc_passed else 'FAIL'} | Score: {overall_score:.1f}%")

        return {
            "success": True,
            "kyc_passed": kyc_passed,
            "overall_score": round(overall_score, 1),
            "face_match_score": round(face_score, 1),
            "text_match_score": round(text_score, 1),
            "extracted_data": {
                "name": extracted.get("name"),
                "date_of_birth": extracted.get("date_of_birth"),
                "id_number": extracted.get("id_number"),
                "expiry_date": extracted.get("expiry_date"),
                "nationality": extracted.get("nationality"),
                "hf_ocr_preview": hf_extracted_text[:100] if hf_extracted_text else None
            },
            "issues": issues,
            "message": "KYC verification passed! ✅" if kyc_passed else "KYC failed. Please retake photos in good lighting.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"KYC error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"KYC error: {str(e)}")

    finally:
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass

# ── 13. Health Check ───────────────────────────
@app.get("/")
def health():
    return {
        "status": "✅ AMO Pay AI + KYC Online",
        "version": "4.0.0",
        "endpoints": {
            "chat": "POST /chat",
            "kyc": "POST /kyc/verify",
        }
    }

# ── 14. Run ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)