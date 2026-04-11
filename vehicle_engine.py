"""
vehicle_engine.py – Person and vehicle detection with YOLO, plus license plate OCR.

Uses YOLOv8n (nano model) which is very fast even on CPU (~50ms/frame).
EasyOCR reads license plates from cropped vehicle ROIs.

Supports Argentine plate formats:
  - Old (pre-2016):  ABC 123   (3 letters + 3 digits)
  - Mercosur (2016+): AB 123 CD (2 letters + 3 digits + 2 letters)
"""
import re
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config import (
    YOLO_MODEL, YOLO_CONF, YOLO_CLASSES, VEHICLE_CLASSES,
    PLATE_MIN_CONF, OCR_LANGS,
    PLATE_REGEX_OLD, PLATE_REGEX_MERCO, PLATE_MIN_LENGTH, PLATE_MAX_LENGTH,
)
import api_client

# ─── Lazy imports (heavy models loaded once on first use) ────────────────────
_yolo = None
_ocr = None


def _get_yolo():
    global _yolo
    if _yolo is None:
        try:
            # PyTorch 2.6+ defaults weights_only=True which breaks YOLO loading.
            # Patch torch.load to force weights_only=False for model loading.
            import torch
            _original_load = torch.load
            def _patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _original_load(*args, **kwargs)
            torch.load = _patched_load

            from ultralytics import YOLO
            _yolo = YOLO(YOLO_MODEL)
            logger.info(f"YOLOv8 loaded: {YOLO_MODEL}")
        except Exception as e:
            logger.error(f"YOLO not available: {e}")
    return _yolo


def _get_ocr():
    global _ocr
    if _ocr is None:
        try:
            import easyocr
            _ocr = easyocr.Reader(OCR_LANGS, gpu=False, verbose=False)
            logger.info("EasyOCR initialized")
        except Exception as e:
            logger.warning(f"EasyOCR not available – plate reading disabled: {e}")
    return _ocr


# ─── Plate helpers ───────────────────────────────────────────────────────────

# Common OCR misreads — map wrong chars to correct ones depending on context
_LETTER_FIXES = str.maketrans("0185", "OIBS")  # digits that look like letters
_DIGIT_FIXES  = str.maketrans("OIBS", "0185")  # letters that look like digits


def _normalize_plate(raw: str) -> str:
    """Remove spaces/special chars, convert to uppercase, keep alphanumerics."""
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def _detect_plate_format(normalized: str) -> Optional[str]:
    """
    Detect if a normalized plate matches Argentine formats.
    Returns: 'old' | 'mercosur' | None
    """
    if len(normalized) == 6 and PLATE_REGEX_OLD.match(normalized):
        return "old"
    if len(normalized) == 7 and PLATE_REGEX_MERCO.match(normalized):
        return "mercosur"
    return None


def _fix_ocr_errors(normalized: str) -> str:
    """
    Apply context-aware fixes for common OCR misreads on Argentine plates.

    Old format: LLL DDD (positions 0-2 = letters, 3-5 = digits)
    Mercosur:   LL DDD LL (positions 0-1 = letters, 2-4 = digits, 5-6 = letters)
    """
    if len(normalized) == 6:
        # Try old format: first 3 should be letters, last 3 digits
        letters = normalized[:3].translate(_LETTER_FIXES)
        digits  = normalized[3:].translate(_DIGIT_FIXES)
        candidate = letters + digits
        if PLATE_REGEX_OLD.match(candidate):
            return candidate

    elif len(normalized) == 7:
        # Try Mercosur: pos 0-1 letters, 2-4 digits, 5-6 letters
        p01 = normalized[0:2].translate(_LETTER_FIXES)
        p24 = normalized[2:5].translate(_DIGIT_FIXES)
        p56 = normalized[5:7].translate(_LETTER_FIXES)
        candidate = p01 + p24 + p56
        if PLATE_REGEX_MERCO.match(candidate):
            return candidate

    return normalized


def _format_plate(normalized: str) -> str:
    """
    Format a normalized plate string for display.
      ABC123  → 'ABC 123'
      AB123CD → 'AB 123 CD'
    Falls back to inserting a space every 3 chars.
    """
    fmt = _detect_plate_format(normalized)
    if fmt == "old":
        return f"{normalized[:3]} {normalized[3:]}"
    elif fmt == "mercosur":
        return f"{normalized[:2]} {normalized[2:5]} {normalized[5:]}"
    # Fallback: split at 3
    if len(normalized) >= 6:
        return f"{normalized[:3]} {normalized[3:]}"
    return normalized


def _preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    """
    Preprocess a cropped vehicle/plate image for better OCR results.
    Applies grayscale → CLAHE contrast → bilateral filter → threshold.
    """
    if crop.size == 0:
        return crop

    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    # CLAHE (adaptive histogram equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Bilateral filter: reduces noise while keeping edges sharp
    filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)

    return filtered


# ─── Known plate registry ────────────────────────────────────────────────────

class VehicleEngine:
    def __init__(self):
        self._known_plates: dict[str, dict] = {}  # normalized_plate -> vehicle_data
        self._last_loaded: float = 0.0

    def reload_if_needed(self) -> None:
        from config import FACE_REFRESH_SEC
        if time.time() - self._last_loaded < FACE_REFRESH_SEC:
            return
        vehicles = api_client.get_authorized_vehicles()
        self._known_plates = {
            _normalize_plate(v["plate"]): v
            for v in vehicles
        }
        self._last_loaded = time.time()
        logger.info(f"Loaded {len(self._known_plates)} authorized plates")

    # ─── Detection ───────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLO on a frame. Returns list of detections (persons + vehicles).
        For vehicles, also runs plate OCR.
        """
        yolo = _get_yolo()
        if yolo is None:
            return []

        self.reload_if_needed()
        results = []

        try:
            predictions = yolo.predict(frame, conf=YOLO_CONF, classes=list(YOLO_CLASSES.keys()), verbose=False)
        except Exception as e:
            logger.error(f"YOLO prediction failed: {e}")
            return []

        for pred in predictions:
            for box in pred.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_name = YOLO_CLASSES.get(cls_id, "unknown")

                if cls_id == 0:
                    # Person detected — face engine will handle recognition separately
                    results.append({
                        "type": "person",
                        "label": "Persona",
                        "authorized": False,
                        "confidence": round(conf * 100, 1),
                        "vehicle_id": None,
                        "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                    })

                elif cls_id in VEHICLE_CLASSES:
                    # Vehicle – try to read plate
                    plate_result = self._read_plate(frame, x1, y1, x2, y2)

                    results.append({
                        "type": "vehicle",
                        "label": plate_result["formatted"] or class_name.capitalize(),
                        "authorized": plate_result["authorized"],
                        "confidence": round(conf * 100, 1),
                        "vehicle_id": plate_result["vehicle_data"].get("id") if plate_result["vehicle_data"] else None,
                        "gate_access": plate_result["vehicle_data"].get("gateAccess", False) if plate_result["vehicle_data"] else False,
                        "vehicle_class": class_name,
                        "plate_raw": plate_result["raw"],
                        "plate_format": plate_result["format"],
                        "plate_confidence": plate_result["ocr_confidence"],
                        "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                    })

        return results

    def _read_plate(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> dict:
        """
        Crop vehicle ROI and run OCR to find license plate text.
        Returns dict with plate info (formatted, raw, format, authorized, vehicle_data, ocr_confidence).
        """
        empty = {
            "formatted": None, "raw": None, "format": None,
            "authorized": False, "vehicle_data": None, "ocr_confidence": 0.0,
        }

        ocr = _get_ocr()
        if ocr is None:
            return empty

        # ── Crop with padding ────────────────────────────────────────────
        h, w = frame.shape[:2]
        pad_x = (x2 - x1) // 8
        pad_y = (y2 - y1) // 8
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return empty

        # ── Focus on the lower third of the vehicle (where plates usually are)
        plate_region_y = int(crop.shape[0] * 0.5)
        plate_crop = crop[plate_region_y:, :]
        if plate_crop.size == 0:
            plate_crop = crop

        # ── Preprocess for better OCR ────────────────────────────────────
        preprocessed = _preprocess_plate_crop(plate_crop)

        # ── Run OCR on both preprocessed (grayscale) and original color ──
        candidates = []

        for img_input in [preprocessed, plate_crop]:
            try:
                ocr_results = ocr.readtext(
                    img_input,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    paragraph=False,
                )
            except Exception as e:
                logger.warning(f"OCR error: {e}")
                continue

            for (_bbox, text, prob) in ocr_results:
                if prob < PLATE_MIN_CONF * 0.8:  # slightly lower threshold for candidates
                    continue
                normalized = _normalize_plate(text)
                if len(normalized) < PLATE_MIN_LENGTH or len(normalized) > PLATE_MAX_LENGTH:
                    continue
                candidates.append((normalized, prob))

        if not candidates:
            return empty

        # ── Score and rank candidates ────────────────────────────────────
        best = None
        best_score = -1

        for normalized, prob in candidates:
            # Apply OCR error correction
            fixed = _fix_ocr_errors(normalized)
            fmt = _detect_plate_format(fixed)

            # Score: OCR confidence + format bonus + authorized bonus
            score = prob
            if fmt is not None:
                score += 0.15  # bonus for valid Argentine format
            vehicle_data = self._known_plates.get(fixed)
            if vehicle_data is not None:
                score += 0.10  # bonus for known plate

            if score > best_score:
                best_score = score
                best = {
                    "formatted": _format_plate(fixed),
                    "raw": fixed,
                    "format": fmt,
                    "authorized": vehicle_data is not None,
                    "vehicle_data": vehicle_data,
                    "ocr_confidence": round(prob * 100, 1),
                }

        if best and best_score >= PLATE_MIN_CONF:
            logger.debug(
                f"Plate read: {best['formatted']} (format={best['format']}, "
                f"conf={best['ocr_confidence']}%, auth={best['authorized']})"
            )
            return best

        return empty
