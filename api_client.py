"""
api_client.py – HTTP client that talks to the NestJS backend.

The Python AI service uses an API key (X-AI-Key header) instead of JWT
because it's an internal service, not a human user.
"""
import httpx
from loguru import logger
from config import BACKEND_URL, AI_API_KEY

_HEADERS = {"X-AI-Key": AI_API_KEY, "Content-Type": "application/json"}
_TIMEOUT = httpx.Timeout(10.0)


def _client() -> httpx.Client:
    return httpx.Client(base_url=BACKEND_URL, headers=_HEADERS, timeout=_TIMEOUT)


# ─── Cameras ────────────────────────────────────────────────────────────────

def get_cameras() -> list[dict]:
    """Return all cameras with their RTSP credentials from the backend."""
    try:
        with _client() as c:
            r = c.get("/cameras")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.warning(f"Could not fetch cameras from backend: {e}")
        return []


# ─── Face encodings ─────────────────────────────────────────────────────────

def get_authorized_faces() -> list[dict]:
    """Return faces that have a photo available for encoding."""
    try:
        with _client() as c:
            r = c.get("/ai/faces")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.warning(f"Could not fetch faces: {e}")
        return []


def get_face_photo(face_id: str) -> bytes | None:
    """Download photo bytes for a given face ID."""
    try:
        with httpx.Client(base_url=BACKEND_URL, headers={"X-AI-Key": AI_API_KEY}, timeout=_TIMEOUT) as c:
            r = c.get(f"/ai/faces/{face_id}/photo")
            r.raise_for_status()
            return r.content
    except Exception:
        return None


def get_authorized_vehicles() -> list[dict]:
    """Return all authorized vehicles (plate + metadata)."""
    try:
        with _client() as c:
            r = c.get("/ai/vehicles")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.warning(f"Could not fetch vehicles: {e}")
        return []


# ─── Report detection ────────────────────────────────────────────────────────

def report_detection(payload: dict, snapshot_path: str | None = None) -> bool:
    """
    POST a detection event to the backend.
    If a snapshot_path is provided, it is uploaded as multipart form.
    """
    import json

    try:
        if snapshot_path:
            # Convert values to proper strings for multipart form data
            # Nested dicts must be JSON-serialized (not Python str())
            form_data = {}
            for k, v in payload.items():
                if v is None:
                    continue
                if isinstance(v, dict):
                    form_data[k] = json.dumps(v)
                elif isinstance(v, bool):
                    form_data[k] = str(v).lower()  # "true"/"false" not "True"/"False"
                else:
                    form_data[k] = str(v)

            with open(snapshot_path, "rb") as img_file:
                with httpx.Client(base_url=BACKEND_URL, timeout=_TIMEOUT) as c:
                    r = c.post(
                        "/ai/report",
                        data=form_data,
                        files={"snapshot": ("snapshot.jpg", img_file, "image/jpeg")},
                        headers={"X-AI-Key": AI_API_KEY},
                    )
        else:
            with _client() as c:
                r = c.post("/ai/report", json=payload)

        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to report detection: {e}")
        return False
