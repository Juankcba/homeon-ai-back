"""
tuya_api.py – HTTP endpoints for local Tuya device control.

Runs as a lightweight Flask/threading HTTP server alongside the main
detection loop. The NestJS backend calls these endpoints to control
Tuya devices locally (arm/disarm alarm, read status, etc.).

Endpoints:
  POST /tuya/status     {device_id, local_key, ip}       → device DPs
  POST /tuya/scan       {device_id, local_key, ip}       → scan + identify DPs
  POST /tuya/mode       {device_id, local_key, ip, mode} → set alarm mode
  POST /tuya/siren      {device_id, local_key, ip, active} → siren control
  GET  /tuya/health                                       → service status
"""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from loguru import logger
from config import AI_API_KEY

try:
    from tuya_local import get_controller, TINYTUYA_AVAILABLE
except ImportError:
    TINYTUYA_AVAILABLE = False

TUYA_API_PORT = 5001


class TuyaHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for Tuya local control."""

    def log_message(self, format, *args):
        # Suppress default HTTP logs; we use loguru
        pass

    def _check_auth(self) -> bool:
        key = self.headers.get("X-AI-Key", "")
        if key != AI_API_KEY:
            self._respond(401, {"error": "Unauthorized"})
            return False
        return True

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _respond(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/tuya/health":
            self._respond(200, {
                "status": "ok",
                "tinytuya_available": TINYTUYA_AVAILABLE,
            })
            return
        self._respond(404, {"error": "Not found"})

    def do_POST(self):
        if not self._check_auth():
            return

        if not TINYTUYA_AVAILABLE:
            self._respond(503, {"error": "tinytuya not installed"})
            return

        body = self._read_body()
        device_id = body.get("device_id", "")
        local_key = body.get("local_key", "")
        ip = body.get("ip", "")
        version = body.get("version", "3.3")

        if not device_id or not local_key or not ip:
            self._respond(400, {"error": "Missing device_id, local_key, or ip"})
            return

        try:
            ctrl = get_controller(device_id, local_key, ip, version)

            if self.path == "/tuya/status":
                result = ctrl.get_status()
                self._respond(200, result)

            elif self.path == "/tuya/scan":
                result = ctrl.scan_dps()
                self._respond(200, result)

            elif self.path == "/tuya/mode":
                mode = body.get("mode", "")
                if mode not in ("arm", "disarm", "home", "sos"):
                    self._respond(400, {"error": f"Invalid mode: {mode}"})
                    return
                result = ctrl.set_mode(mode)
                self._respond(200, {"success": True, "result": result})

            elif self.path == "/tuya/siren":
                active = body.get("active", False)
                result = ctrl.set_siren(active)
                self._respond(200, {"success": True, "result": result})

            else:
                self._respond(404, {"error": "Not found"})

        except Exception as e:
            logger.error(f"Tuya local error: {e}")
            self._respond(500, {"error": str(e)})


def start_tuya_api():
    """Start the Tuya local control HTTP server in a background thread."""
    if not TINYTUYA_AVAILABLE:
        logger.warning("tinytuya not available — local Tuya API disabled")
        return

    def _run():
        server = HTTPServer(("0.0.0.0", TUYA_API_PORT), TuyaHandler)
        logger.info(f"Tuya local API listening on port {TUYA_API_PORT}")
        server.serve_forever()

    thread = threading.Thread(target=_run, daemon=True, name="tuya-api")
    thread.start()
