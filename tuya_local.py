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
    KNOWN_ALARM_DPS = [4, 29, 104]   # alarm/siren active
    KNOWN_VOLUME_DPS = [13, 113]     # alarm volume

    MODE_MAP = {
        "arm": "arm",
        "disarm": "disarm",
        "home": "home",
        "sos": "sos",
    }

    def __init__(self, device_id: str, local_key: str, ip: str, version: str = "3.3"):
        self.device = TuyaLocalDevice(device_id, local_key, ip, version)
        self._mode_dp: Optional[int] = None
        self._alarm_dp: Optional[int] = None
        self._dp_map: dict = {}

    def scan_dps(self) -> dict:
        """Read device and identify which DPs are available."""
        dps = self.device.get_dps()
        self._dp_map = dps

        # Try to identify mode DP (string values like "arm", "disarm")
        for dp in self.KNOWN_MODE_DPS:
            if str(dp) in dps and isinstance(dps[str(dp)], str):
                self._mode_dp = dp
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
        }

    def get_status(self) -> dict:
        """Get current alarm status with human-readable fields."""
        dps = self.device.get_dps()
        self._dp_map = dps

        mode = "unknown"
        alarm_active = False

        # Extract mode
        if self._mode_dp and str(self._mode_dp) in dps:
            mode = dps[str(self._mode_dp)]
        else:
            for dp in self.KNOWN_MODE_DPS:
                val = dps.get(str(dp))
                if isinstance(val, str) and val in ("arm", "disarm", "home", "sos"):
                    mode = val
                    self._mode_dp = dp
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
        if mode not in self.MODE_MAP:
            raise ValueError(f"Invalid mode: {mode}")

        # If we know the mode DP, use it
        if self._mode_dp:
            return self.device.set_dp(self._mode_dp, self.MODE_MAP[mode])

        # Try each known mode DP
        for dp in self.KNOWN_MODE_DPS:
            try:
                result = self.device.set_dp(dp, self.MODE_MAP[mode])
                if result and "Error" not in result:
                    self._mode_dp = dp
                    return result
            except Exception:
                continue

        raise RuntimeError(f"Could not find mode DP for alarm")

    def set_siren(self, active: bool) -> dict:
        """Activate or deactivate the siren."""
        if self._alarm_dp:
            return self.device.set_dp(self._alarm_dp, active)

        for dp in self.KNOWN_ALARM_DPS:
            try:
                result = self.device.set_dp(dp, active)
                if result and "Error" not in result:
                    self._alarm_dp = dp
                    return result
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
