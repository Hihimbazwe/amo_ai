# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AMO PAY — Local AI Model Bootstrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
import sys
import time
import logging
from sentence_transformers import SentenceTransformer
import easyocr
import mediapipe as mp
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Bootstrap")

def download_models():
    print("\n" + "="*50)
    print("🚀 Starting AMO Pay Local AI Model Bootstrap")
    print("This will download all required models for KYC and Chat.")
    print("="*50 + "\n")

    # 1. Sentence Transformers
    try:
        print("📦 [1/4] Downloading SentenceTransformer (all-MiniLM-L6-v2)...")
        SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ SentenceTransformer ready.")
    except Exception as e:
        print(f"❌ Failed to download SentenceTransformer: {e}")

    # 2. YOLOv8
    try:
        print("\n📦 [2/4] Downloading YOLOv8n Weights...")
        YOLO('yolov8n.pt')
        print("✅ YOLOv8 weights ready.")
    except Exception as e:
        print(f"❌ Failed to download YOLO weights: {e}")

    # 3. EasyOCR
    try:
        print("\n📦 [3/4] Downloading EasyOCR Models (English)...")
        easyocr.Reader(['en'], gpu=False)
        print("✅ EasyOCR models ready.")
    except Exception as e:
        print(f"❌ Failed to download EasyOCR models: {e}")

    # 4. MediaPipe
    try:
        print("\n📦 [4/4] Initializing MediaPipe FaceMesh...")
        mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("✅ MediaPipe ready.")
    except Exception as e:
        print(f"❌ Failed to initialize MediaPipe: {e}")

    print("\n" + "="*50)
    print("🎉 All models are now cached locally!")
    print("You can now start the server with: python server.py")
    print("="*50 + "\n")

if __name__ == "__main__":
    download_models()
