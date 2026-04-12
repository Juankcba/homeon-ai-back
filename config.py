"""
config.py – Loads all settings from environment variables.
All defaults are intentionally conservative (low CPU, safe thresholds).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Backend API ────────────────────────────────────────────────────────────
BACKEND_URL       = os.getenv("BACKEND_URL", "http://127.0.0.1:3005")
AI_API_KEY        = os.getenv("AI_API_KEY", "changeme-ai-secret")

# ─── Cameras ────────────────────────────────────────────────────────────────
# Comma-separated RTSP URLs. The NestJS backend is the source of truth,
# but we also support env-based overrides for quick local testing.
# Format: rtsp://user:pass@ip:554/stream1
RTSP_OVERRIDE     = os.getenv("RTSP_OVERRIDE", "")   # optional manual list

# ─── Detection pipeline ─────────────────────────────────────────────────────
FRAME_INTERVAL    = float(os.getenv("FRAME_INTERVAL", "30"))    # seconds between captures
YOLO_MODEL        = os.getenv("YOLO_MODEL", "yolov8n.pt")       # nano for speed
YOLO_CONF         = float(os.getenv("YOLO_CONF", "0.45"))       # confidence threshold
FACE_TOLERANCE    = float(os.getenv("FACE_TOLERANCE", "0.55"))  # lower = stricter
PLATE_MIN_CONF    = float(os.getenv("PLATE_MIN_CONF", "0.60"))  # OCR min confidence

# ─── Snapshots ──────────────────────────────────────────────────────────────
SNAPSHOT_DIR      = os.getenv("SNAPSHOT_DIR", "/snapshots")
MAX_SNAPSHOT_DAYS = int(os.getenv("MAX_SNAPSHOT_DAYS", "3"))    # auto cleanup

# ─── YOLO classes of interest ───────────────────────────────────────────────
# COCO class IDs: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
YOLO_CLASSES      = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES   = {2, 3, 5, 7}

# ─── Face encoding refresh ──────────────────────────────────────────────────
FACE_REFRESH_SEC  = int(os.getenv("FACE_REFRESH_SEC", "300"))   # reload DB faces every 5 min

# ─── OCR language ───────────────────────────────────────────────────────────
OCR_LANGS         = os.getenv("OCR_LANGS", "es,en").split(",")  # EasyOCR languages

# ─── Plate format patterns (Argentina) ─────────────────────────────────────
# Old format (pre-2016): ABC 123  (3 letters + 3 digits)
# New Mercosur (2016+):  AB 123 CD (2 letters + 3 digits + 2 letters)
import re
PLATE_REGEX_OLD   = re.compile(r'^[A-Z]{3}\d{3}$')         # ABC123
PLATE_REGEX_MERCO = re.compile(r'^[A-Z]{2}\d{3}[A-Z]{2}$') # AB123CD
PLATE_MIN_LENGTH  = 6
PLATE_MAX_LENGTH  = 7
