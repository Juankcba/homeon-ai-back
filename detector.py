"""
detector.py – Main detection pipeline.

For each camera frame:
1. Run YOLO → detect persons and vehicles
2. For each person region → run face_engine to identify
3. For each vehicle → already has plate from vehicle_engine
4. Save annotated snapshot to disk
5. Report each detection to NestJS backend
"""
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from config import SNAPSHOT_DIR, MAX_SNAPSHOT_DAYS
from face_engine import FaceEngine
from vehicle_engine import VehicleEngine
import api_client


class Detector:
    def __init__(self):
        self.face_engine = FaceEngine()
        self.vehicle_engine = VehicleEngine()
        Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)

    def process_frame(self, camera_id: str, camera_name: str, frame: np.ndarray) -> list[dict]:
        """
        Full pipeline: run YOLO + face recognition on a single frame.
        Returns list of detection dicts.
        """
        all_detections = []

        # ── 1. Vehicle + person detection via YOLO ───────────────────────
        yolo_results = self.vehicle_engine.detect(frame)

        # ── 2. Face recognition ──────────────────────────────────────────
        face_results = self.face_engine.detect(frame)

        # ── 3. Merge: if a YOLO person bbox overlaps a face bbox, upgrade it
        # (avoids double-counting person + face detection)
        face_bboxes = [r["bbox"] for r in face_results]
        filtered_yolo = []
        for det in yolo_results:
            if det["type"] == "person":
                # Check if any face was detected in approximately the same area
                overlaps = any(
                    _boxes_overlap(det["bbox"], fb, threshold=0.3)
                    for fb in face_bboxes
                )
                if overlaps:
                    continue  # face_engine will handle this one
            filtered_yolo.append(det)

        all_detections = face_results + filtered_yolo

        # ── 4. Save snapshot + report each detection ─────────────────────
        for det in all_detections:
            snapshot_path = self._save_snapshot(frame, camera_id, det)
            payload = {
                "type": det["type"],
                "label": det["label"],
                "cameraId": camera_id,
                "cameraName": camera_name,
                "confidence": det["confidence"],
                "authorized": det.get("authorized", False),
                "matchedFaceId": det.get("face_id"),
                "matchedVehicleId": det.get("vehicle_id"),
                "boundingBox": det.get("bbox"),
                "metadata": {
                    "gateAccess": det.get("gate_access", False),
                    "vehicleClass": det.get("vehicle_class"),
                },
            }
            api_client.report_detection(payload, snapshot_path)
            logger.info(
                f"[{camera_name}] {det['type'].upper()}: {det['label']} "
                f"({det['confidence']}%) {'✅' if det.get('authorized') else '⚠️'}"
            )

        return all_detections

    def _save_snapshot(self, frame: np.ndarray, camera_id: str, det: dict) -> Optional[str]:
        """Crop the detected region + some context, save as JPEG."""
        try:
            date_str = datetime.utcnow().strftime("%Y/%m/%d")
            folder = Path(SNAPSHOT_DIR) / camera_id / date_str
            folder.mkdir(parents=True, exist_ok=True)

            det_type = det["type"]
            label_safe = det["label"].replace(" ", "_")
            filename = f"{det_type}_{label_safe}_{uuid.uuid4().hex[:8]}.jpg"
            path = str(folder / filename)

            # Draw bounding box on a copy of the frame
            annotated = frame.copy()
            bbox = det.get("bbox")
            if bbox:
                color = (0, 200, 0) if det.get("authorized") else (0, 0, 220)
                cv2.rectangle(
                    annotated,
                    (bbox["x"], bbox["y"]),
                    (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                    color, 2
                )
                cv2.putText(
                    annotated,
                    f"{det['label']} {det['confidence']}%",
                    (bbox["x"], max(bbox["y"] - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

            cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return path
        except Exception as e:
            logger.warning(f"Could not save snapshot: {e}")
            return None

    def cleanup_old_snapshots(self) -> None:
        """Delete snapshots older than MAX_SNAPSHOT_DAYS."""
        cutoff = time.time() - MAX_SNAPSHOT_DAYS * 86400
        root = Path(SNAPSHOT_DIR)
        deleted = 0
        for f in root.rglob("*.jpg"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                deleted += 1
        if deleted:
            logger.info(f"Cleaned up {deleted} old snapshots")


def _boxes_overlap(a: dict, b: dict, threshold: float = 0.3) -> bool:
    """Check if two bounding boxes overlap by more than `threshold` IoU."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = a["width"] * a["height"]
    b_area = b["width"] * b["height"]
    union_area = a_area + b_area - inter_area

    return (inter_area / union_area) >= threshold if union_area > 0 else False
