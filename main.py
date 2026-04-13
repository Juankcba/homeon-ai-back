"""
main.py – HomeOn AI Engine entry point.

Workflow:
1. Fetch camera list from NestJS backend
2. Start an RTSP reader thread per camera
3. Every FRAME_INTERVAL seconds, grab the latest frame from each camera
4. Run the detection pipeline on each frame
5. Report results to NestJS backend
6. Cleanup old snapshots once a day
"""
import os
import signal
import sys
import time
from datetime import datetime

import schedule
from loguru import logger

from config import FRAME_INTERVAL, RTSP_OVERRIDE
from camera_reader import RTSPCamera, build_rtsp_url
from detector import Detector
import api_client

# ─── Logging ────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")

# ─── Global state ───────────────────────────────────────────────────────────
cameras: list[RTSPCamera] = []
detector: Detector | None = None

# Runtime flag – polled from backend every CONFIG_POLL_INTERVAL seconds.
# When False, detection cycles are skipped (but cameras keep running so we
# can resume instantly).
_detection_enabled: bool = True
_last_config_poll: float = 0.0
CONFIG_POLL_INTERVAL = 10.0  # seconds


def refresh_engine_config() -> None:
    """Poll backend for runtime config (detectionEnabled)."""
    global _detection_enabled, _last_config_poll
    cfg = api_client.get_engine_config()
    new_enabled = bool(cfg.get("detectionEnabled", True))
    if new_enabled != _detection_enabled:
        state = "ENABLED" if new_enabled else "DISABLED"
        logger.info(f"AI detection {state} via backend config")
    _detection_enabled = new_enabled
    _last_config_poll = time.time()


def init_cameras() -> None:
    """Fetch cameras from backend and start RTSP reader threads."""
    global cameras

    # Stop existing cameras
    for cam in cameras:
        cam.stop()
    cameras.clear()

    if RTSP_OVERRIDE:
        # Manual override: comma-separated RTSP URLs
        urls = [u.strip() for u in RTSP_OVERRIDE.split(",") if u.strip()]
        for i, url in enumerate(urls):
            cam = RTSPCamera(camera_id=str(i), name=f"Camera {i+1}", rtsp_url=url)
            cam.start()
            cameras.append(cam)
    else:
        # Fetch from backend API
        cam_data = api_client.get_cameras()
        if not cam_data:
            logger.warning("No cameras from backend – waiting for next retry")
            return

        for c in cam_data:
            if c.get("status") == "offline":
                continue
            rtsp = build_rtsp_url(c)
            cam = RTSPCamera(camera_id=c["id"], name=c.get("name", c["id"]), rtsp_url=rtsp)
            cam.start()
            cameras.append(cam)

    logger.info(f"Initialized {len(cameras)} camera(s)")


def run_detection_cycle() -> None:
    """Grab a frame from each camera and run the full detection pipeline."""
    if not cameras:
        return

    now = datetime.utcnow().strftime("%H:%M:%S")
    for cam in cameras:
        frame = cam.get_frame()
        if frame is None:
            if not cam.is_online():
                logger.debug(f"[{cam.name}] No frame available – skipping")
            continue

        detections = detector.process_frame(cam.camera_id, cam.name, frame)
        count = len(detections)
        if count:
            logger.info(f"[{cam.name}] {now} → {count} detection(s)")


def cleanup_job() -> None:
    if detector:
        detector.cleanup_old_snapshots()


def shutdown(sig, frame) -> None:
    logger.info("Shutting down AI engine…")
    for cam in cameras:
        cam.stop()
    sys.exit(0)


def main() -> None:
    global detector

    # Ensure all files/dirs created by this process are world-writable.
    # The /snapshots volume is shared with the NestJS backend container.
    os.umask(0o000)

    logger.info("=" * 50)
    logger.info("  HomeOn AI Engine  –  starting up")
    logger.info("=" * 50)

    detector = Detector()

    # Clean old snapshots immediately at startup
    detector.cleanup_old_snapshots()

    # Initial camera load
    init_cameras()

    # Retry camera load every 60 s if none loaded
    schedule.every(60).seconds.do(lambda: init_cameras() if not cameras else None)

    # Snapshot cleanup every 6 hours (prevents disk from filling up)
    schedule.every(6).hours.do(cleanup_job)

    # Reload cameras once per hour (picks up new cameras added via UI)
    schedule.every(1).hours.do(init_cameras)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start Tuya local control API (for alarm arm/disarm via LAN)
    try:
        from tuya_api import start_tuya_api
        start_tuya_api()
    except Exception as e:
        logger.warning(f"Tuya local API not started: {e}")

    logger.info(f"Detection cycle every {FRAME_INTERVAL}s – waiting for first frames…")
    # Give cameras 5 s to connect before first detection
    time.sleep(5)

    # Prime the config flag before entering the loop
    refresh_engine_config()

    last_detection = 0.0
    while True:
        schedule.run_pending()
        now = time.time()

        # Refresh runtime config periodically (cheap GET)
        if now - _last_config_poll >= CONFIG_POLL_INTERVAL:
            refresh_engine_config()

        if now - last_detection >= FRAME_INTERVAL:
            if _detection_enabled:
                run_detection_cycle()
            last_detection = now

        time.sleep(0.5)


if __name__ == "__main__":
    main()
