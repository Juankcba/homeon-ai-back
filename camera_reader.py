"""
camera_reader.py – RTSP stream reader using OpenCV.

Opens an RTSP connection, grabs a single frame on demand.
Uses a background thread to keep the buffer fresh and avoid
accumulating stale frames (common RTSP issue with OpenCV).
"""
import threading
import time
from typing import Optional

import cv2
import numpy as np
from loguru import logger


class RTSPCamera:
    """
    Maintains a live RTSP connection.
    A background thread continuously reads frames; .get_frame() returns the latest.
    """

    def __init__(self, camera_id: str, name: str, rtsp_url: str):
        self.camera_id = camera_id
        self.name = name
        self.rtsp_url = rtsp_url
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_frame_time: float = 0.0
        self._connect_attempts = 0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name=f"rtsp-{self.name}")
        self._thread.start()
        logger.info(f"[{self.name}] RTSP reader started")

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame, or None if not available."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_online(self) -> bool:
        return (
            self._frame is not None
            and time.time() - self._last_frame_time < 30
        )

    def _connect(self) -> bool:
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = self._cap.isOpened()
        if ok:
            logger.info(f"[{self.name}] Connected to RTSP stream")
            self._connect_attempts = 0
        else:
            self._connect_attempts += 1
            logger.warning(f"[{self.name}] RTSP connection failed (attempt {self._connect_attempts})")
        return ok

    def _reader_loop(self) -> None:
        """Continuously grab frames; reconnect on failure with exponential backoff."""
        while self._running:
            if not self._cap or not self._cap.isOpened():
                if not self._connect():
                    sleep_s = min(30, 2 ** self._connect_attempts)
                    time.sleep(sleep_s)
                    continue

            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                self._last_frame_time = time.time()
            else:
                logger.warning(f"[{self.name}] Frame read failed – reconnecting")
                self._cap.release()
                self._cap = None
                time.sleep(2)


def build_rtsp_url(camera: dict) -> str:
    """
    Build an RTSP URL from NestJS camera data.
    Expected fields: rtspUrl (preferred) OR ip/rtspUsername/rtspPassword.
    """
    if camera.get("rtspUrl"):
        return camera["rtspUrl"]

    user = camera.get("rtspUsername") or camera.get("username", "admin")
    password = camera.get("rtspPassword") or camera.get("password", "")
    host = camera.get("ip") or camera.get("host", "")
    port = camera.get("port", 554)
    return f"rtsp://{user}:{password}@{host}:{port}/stream1"
