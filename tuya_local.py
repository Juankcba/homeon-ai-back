"""
tuya_local.py – Local LAN control for Tuya / Smart Life devices.

Uses tinytuya to communicate directly with devices on the local network,
bypassing the Tuya Cloud API. This is faster, more reliable, and works
even when internet is down.

For alarm gateways (category 'wfcon'), the Cloud API returns empty status/functions.
Local control via tinytuya can read DPs directly and send commands.

Common alarm gateway DPs (varies by model):
  DP 1  → master_mode: "arm" / "disarm" / "home" / "sos"
  DP 2  → delay_set (entry/exit delay in seconds)
  DP 4  → alarm_state: true/false (siren active)
  DP 13 → alarm_volume: "low" / "medium" / "high" / "mute"
  DP 15 → alarm_ringtone
  DP 29 → switch_alarm: true/false
"""
import json
from typing import Optional
from loguru import logger

try:
    import tinytuya
    TINYTUYA_AVAILABLE = True
except ImportError:
    logger.warning("tinytuya not installed — local Tuya control disabled")
    TINYTUYA_AVAILABLE = False


class TuyaLocalDevice:
    """Wrapper around a tinytuya device connection."""

    def __init__(self, device_id: str, local_key: str, ip: str, version: str = "3.3"):
        self.device_id = device_id
        self.local_key = local_key
        self.ip = ip
        self.version = version
        self._device: Optional[object] = None

    def _connect(self) -> "tinytuya.Device":
        if not TINYTUYA_AVAILABLE:
            raise RuntimeError("tinytuya not installed")

        d = tinytuya.Device(
            dev_id=self.device_id,
            address=self.ip,
            local_key=self.local_key,
            version=self.version,
        )
        d.set_socketPersistent(False)
        self._device = d
        return d

    def get_status(self) -> dict:
        """Read all DPs from the device."""
        d = self._connect()
        data = d.status()
        if "Error" in data:
            logger.error(f"Tuya local status error for {self.ip}: {data}")
            # Try different protocol versions
            for v in ["3.4", "3.3", "3.1"]:
                if v == self.version:
                    continue
                d.set_version(float(v))
                data = d.status()
                if "Error" not in data:
                    self.version = v
                    logger.info(f"Tuya device {self.ip} works with version {v}")
                    break

        return data

    def get_dps(self) -> dict:
        """Return just the DPs dict."""
        status = self.get_status()
        return status.get("dps", {})

    def set_dp(self, dp_index: int, value) -> dict:
        """Set a single DP value."""
        d = self._connect()
        result = d.set_value(dp_index, value)
        logger.info(f"Tuya local set DP {dp_index}={value} on {self.ip}: {result}")
        return result or {}

    def set_multiple_dps(self, dps: dict) -> dict:
        """Set multiple DPs at once."""
        d = self._connect()
        result = d.set_multiple_values(dps)
        logger.info(f"Tuya local set DPs {dps} on {self.ip}: {result}")
        return result or {}


class TuyaAlarmController:
    """
    High-level alarm control using local Tuya communication.

    Since different alarm models use different DP mappings, this class
    auto-detects the correct DPs on first connection by reading the
    device status and matching known patterns.
    """

    # Known DP mappings for alarm gateways
    KNOWN_MODE_DPS = [1, 101]        # master_mode
    KNOWN_ALARM_DPS = [4, 29, 104, 103]   # alarm/siren active
    KNOWN_VOLUME_DPS = [13, 113]     # alarm volume

    # Mode translation: different models use different value formats
    # Some use strings "arm"/"disarm", others use numeric strings "0"/"1"/"2"
    # We normalize everything to our standard: arm, disarm, home, sos

    # Numeric string → standard mode name
    NUMERIC_TO_MODE = {
        "0": "disarm",
        "1": "home",
        "2": "arm",
        "3": "sos",
    }

    # Standard mode name → numeric string (for devices that use numbers)
    MODE_TO_NUMERIC = {
        "disarm": "0",
        "home": "1",
        "arm": "2",
        "sos": "3",
    }

    # Standard mode name → string value (for devices that use words)
    MODE_TO_STRING = {
        "arm": "arm",
        "disarm": "disarm",
        "home": "home",
        "sos": "sos",
    }

    def __init__(self, device_id: str, local_key: str, ip: str, version: str = "3.3"):
        self.device = TuyaLocalDevice(device_id, local_key, ip, version)
        self._mode_dp: Optional[int] = None
        self._alarm_dp: Optional[int] = None
        self._uses_numeric_modes: bool = True  # default: assume numeric
        self._dp_map: dict = {}

    def scan_dps(self) -> dict:
        """Read device and identify which DPs are available."""
        dps = self.device.get_dps()
        self._dp_map = dps

        # Try to identify mode DP (string values)
        for dp in self.KNOWN_MODE_DPS:
            val = dps.get(str(dp))
            if isinstance(val, str):
                self._mode_dp = dp
                # Detect if it uses numeric ("0","1","2") or word ("arm","disarm") values
                self._uses_numeric_modes = val in self.NUMERIC_TO_MODE
                break

        # Try to identify alarm/siren DP (boolean)
        for dp in self.KNOWN_ALARM_DPS:
            if str(dp) in dps and isinstance(dps[str(dp)], bool):
                self._alarm_dp = dp
                break

        return {
            "dps": dps,
            "detected_mode_dp": self._mode_dp,
            "detected_alarm_dp": self._alarm_dp,
            "uses_numeric_modes": self._uses_numeric_modes,
        }

    def _raw_mode_to_standard(self, raw: str) -> str:
        """Convert device raw mode value to standard name."""
        if raw in self.NUMERIC_TO_MODE:
            return self.NUMERIC_TO_MODE[raw]
        if raw in ("arm", "disarm", "home", "sos"):
            return raw
        return "unknown"

    def _standard_mode_to_raw(self, mode: str) -> str:
        """Convert standard mode name to device raw value."""
        if self._uses_numeric_modes:
            return self.MODE_TO_NUMERIC.get(mode, "0")
        return self.MODE_TO_STRING.get(mode, mode)

    def get_status(self) -> dict:
        """Get current alarm status with human-readable fields."""
        dps = self.device.get_dps()
        self._dp_map = dps

        mode = "unknown"
        alarm_active = False

        # Extract mode
        if self._mode_dp and str(self._mode_dp) in dps:
            raw = dps[str(self._mode_dp)]
            mode = self._raw_mode_to_standard(str(raw))
            self._uses_numeric_modes = str(raw) in self.NUMERIC_TO_MODE
        else:
            for dp in self.KNOWN_MODE_DPS:
                val = dps.get(str(dp))
                if isinstance(val, str):
                    mode = self._raw_mode_to_standard(val)
                    self._mode_dp = dp
                    self._uses_numeric_modes = val in self.NUMERIC_TO_MODE
                    break

        # Extract alarm state
        if self._alarm_dp and str(self._alarm_dp) in dps:
            alarm_active = bool(dps[str(self._alarm_dp)])
        else:
            for dp in self.KNOWN_ALARM_DPS:
                val = dps.get(str(dp))
                if isinstance(val, bool):
                    alarm_active = val
                    self._alarm_dp = dp
                    break

        return {
            "mode": mode,
            "alarmActive": alarm_active,
            "online": True,
            "dps": dps,
        }

    def set_mode(self, mode: str) -> dict:
        """Set alarm mode: arm, disarm, home, sos."""
        if mode not in self.MODE_TO_NUMERIC:
            raise ValueError(f"Invalid mode: {mode}")

        raw_value = self._standard_mode_to_raw(mode)

        # Auto-scan if mode DP not yet known
        if not self._mode_dp:
            self.scan_dps()

        # If we know the mode DP, use it directly
        if self._mode_dp:
            result = self.device.set_dp(self._mode_dp, raw_value)
            # tinytuya returns None on success (no ack), dict with "Error" on failure
            if isinstance(result, dict) and "Error" in result:
                raise RuntimeError(f"DP {self._mode_dp} error: {result}")
            logger.info(f"Set alarm mode DP {self._mode_dp} = {raw_value} ({mode})")
            return {"success": True, "dp": self._mode_dp, "value": raw_value, "mode": mode}

        # Fallback: try each known mode DP
        for dp in self.KNOWN_MODE_DPS:
            try:
                result = self.device.set_dp(dp, raw_value)
                # None = success, dict with "Error" = failure
                if isinstance(result, dict) and "Error" in result:
                    continue
                self._mode_dp = dp
                logger.info(f"Discovered mode DP {dp}, set to {raw_value} ({mode})")
                return {"success": True, "dp": dp, "value": raw_value, "mode": mode}
            except Exception:
                continue

        raise RuntimeError("Could not find mode DP for alarm")

    def set_siren(self, active: bool) -> dict:
        """Activate or deactivate the siren."""
        if not self._alarm_dp:
            self.scan_dps()

        if self._alarm_dp:
            result = self.device.set_dp(self._alarm_dp, active)
            if isinstance(result, dict) and "Error" in result:
                raise RuntimeError(f"Siren DP {self._alarm_dp} error: {result}")
            logger.info(f"Set siren DP {self._alarm_dp} = {active}")
            return {"success": True, "dp": self._alarm_dp, "active": active}

        for dp in self.KNOWN_ALARM_DPS:
            try:
                result = self.device.set_dp(dp, active)
                if isinstance(result, dict) and "Error" in result:
                    continue
                self._alarm_dp = dp
                logger.info(f"Discovered siren DP {dp}, set to {active}")
                return {"success": True, "dp": dp, "active": active}
            except Exception:
                continue

        raise RuntimeError("Could not find siren DP for alarm")


# ─── Module-level singleton ─────────────────────────────────────────────────

_controllers: dict[str, TuyaAlarmController] = {}


def get_controller(device_id: str, local_key: str, ip: str, version: str = "3.3") -> TuyaAlarmController:
    """Get or create a controller for a device."""
    if device_id not in _controllers:
        _controllers[device_id] = TuyaAlarmController(device_id, local_key, ip, version)
    return _controllers[device_id]
