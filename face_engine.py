"""
face_engine.py – Face detection and recognition.

Uses the `face_recognition` library (dlib under the hood).
Maintains an in-memory registry of known face encodings loaded from the
NestJS backend at startup and refreshed periodically.
"""
import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    logger.warning("face_recognition not available – face matching disabled")
    FACE_RECOGNITION_AVAILABLE = False

from config import FACE_TOLERANCE, FACE_REFRESH_SEC
import api_client


class FaceEngine:
    def __init__(self):
        self._known_encodings: list[np.ndarray] = []
        self._known_names: list[str] = []
        self._known_ids: list[str] = []
        self._known_gate_access: list[bool] = []
        self._last_loaded: float = 0.0

    # ─── Registry ────────────────────────────────────────────────────────

    def reload_if_needed(self) -> None:
        if time.time() - self._last_loaded < FACE_REFRESH_SEC:
            return
        self._load_from_backend()

    def _load_from_backend(self) -> None:
        if not FACE_RECOGNITION_AVAILABLE:
            return

        faces = api_client.get_authorized_faces()
        encodings, names, ids, gates = [], [], [], []

        for face in faces:
            photo_bytes = api_client.get_face_photo(face["id"])
            if not photo_bytes:
                continue
            try:
                img = np.array(Image.open(io.BytesIO(photo_bytes)).convert("RGB"))
                encs = face_recognition.face_encodings(img)
                if encs:
                    encodings.append(encs[0])
                    names.append(face["name"])
                    ids.append(face["id"])
                    gates.append(face.get("gateAccess", False))
            except Exception as e:
                logger.warning(f"Could not encode face for {face.get('name')}: {e}")

        self._known_encodings = encodings
        self._known_names = names
        self._known_ids = ids
        self._known_gate_access = gates
        self._last_loaded = time.time()
        logger.info(f"Loaded {len(encodings)} face encodings from backend")

    # ─── Detection ───────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect faces in a frame.
        Returns a list of dicts with: label, face_id, authorized, confidence, bbox.
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return []

        self.reload_if_needed()

        # Resize for speed (process at 1/4 resolution, then scale bbox back)
        # np.ascontiguousarray is required because dlib's compute_face_descriptor
        # expects C-contiguous memory; sliced views (::2, ::-1) are non-contiguous.
        small = np.ascontiguousarray(frame[::2, ::2, ::-1])  # downsample + BGR→RGB

        locations = face_recognition.face_locations(small, model="hog")
        if not locations:
            return []

        encodings = face_recognition.face_encodings(small, locations)
        results = []

        for enc, loc in zip(encodings, locations):
            top, right, bottom, left = [v * 2 for v in loc]  # scale back up
            label = "Desconocido"
            face_id = None
            authorized = False
            confidence = 0.0
            gate_access = False

            if self._known_encodings:
                distances = face_recognition.face_distance(self._known_encodings, enc)
                best_idx = int(np.argmin(distances))
                best_dist = float(distances[best_idx])
                if best_dist <= FACE_TOLERANCE:
                    label = self._known_names[best_idx]
                    face_id = self._known_ids[best_idx]
                    gate_access = self._known_gate_access[best_idx]
                    authorized = True
                    confidence = round((1 - best_dist) * 100, 1)
                else:
                    confidence = round(max(0, (1 - best_dist) * 100), 1)

            results.append({
                "type": "face",
                "label": label,
                "face_id": face_id,
                "authorized": authorized,
                "confidence": confidence,
                "gate_access": gate_access,
                "bbox": {"x": left, "y": top, "width": right - left, "height": bottom - top},
            })

        return results

    # ─── Crop face from frame ────────────────────────────────────────────

    @staticmethod
    def crop_face(frame: np.ndarray, bbox: dict, padding: int = 30) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x = max(0, bbox["x"] - padding)
        y = max(0, bbox["y"] - padding)
        x2 = min(w, bbox["x"] + bbox["width"] + padding)
        y2 = min(h, bbox["y"] + bbox["height"] + padding)
        crop = frame[y:y2, x:x2]
        return crop if crop.size > 0 else None
