"""
vehicle_engine.py – Person and vehicle detection with YOLO, plus license plate OCR.

Uses YOLOv8n (nano model) which is very fast even on CPU (~50ms/frame).
EasyOCR reads license plates from cropped vehicle ROIs.
"""
import re
import time
from typing import Optional

import numpy as np
from loguru import logger

from config import YOLO_MODEL, YOLO_CONF, YOLO_CLASSES, VEHICLE_CLASSES, PLATE_MIN_CONF, OCR_LANGS
import api_client

# ─── Lazy imports (heavy models loaded once on first use) ────────────────────
_yolo = None
_ocr = None


def _get_yolo():
    global _yolo
    if _yolo is None:
        try:
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


# ─── Known plate registry ────────────────────────────────────────────────────

class VehicleEngine:
    def __init__(self):
        self._known_plates: dict[str, dict] = {}  # normalized_plate -> vehicle_data
        self._last_loaded: float = 0.0

    def _normalize_plate(self, raw: str) -> str:
        """Remove spaces, convert to uppercase, keep alphanumerics."""
        return re.sub(r"[^A-Z0-9]", "", raw.upper())

    def reload_if_needed(self) -> None:
        from config import FACE_REFRESH_SEC
        if time.time() - self._last_loaded < FACE_REFRESH_SEC:
            return
        vehicles = api_client.get_authorized_vehicles()
        self._known_plates = {
            self._normalize_plate(v["plate"]): v
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
                    plate_text, authorized, vehicle_data = self._read_plate(frame, x1, y1, x2, y2)

                    results.append({
                        "type": "vehicle",
                        "label": plate_text or class_name.capitalize(),
                        "authorized": authorized,
                        "confidence": round(conf * 100, 1),
                        "vehicle_id": vehicle_data.get("id") if vehicle_data else None,
                        "gate_access": vehicle_data.get("gateAccess", False) if vehicle_data else False,
                        "vehicle_class": class_name,
                        "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                    })

        return results

    def _read_plate(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int):
        """Crop vehicle ROI and run OCR to find license plate text."""
        ocr = _get_ocr()
        if ocr is None:
            return None, False, None

        # Expand crop slightly for better OCR
        h, w = frame.shape[:2]
        pad_x = (x2 - x1) // 8
        pad_y = (y2 - y1) // 8
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None, False, None

        try:
            ocr_results = ocr.readtext(crop, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
        except Exception as e:
            logger.warning(f"OCR error: {e}")
            return None, False, None

        # Find best plate candidate (short alphanum string with high confidence)
        for (_bbox, text, prob) in sorted(ocr_results, key=lambda x: x[2], reverse=True):
            normalized = self._normalize_plate(text)
            if len(normalized) >= 5 and prob >= PLATE_MIN_CONF:
                # Try to match against authorized plates
                vehicle_data = self._known_plates.get(normalized)
                authorized = vehicle_data is not None
                # Format nicely: "ABC 123" style
                formatted = f"{normalized[:3]} {normalized[3:]}" if len(normalized) >= 6 else normalized
                return formatted, authorized, vehicle_data

        return None, False, None
