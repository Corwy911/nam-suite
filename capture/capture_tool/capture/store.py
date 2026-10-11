"""Calibrations persisted in nam-web/calibration.json (atomic writes). One entry per (kind, device, channel, sample rate):

  kind "out"  Co = dBu at the output for a 0 dBFS sine (multimeter)          -> NAM input_level_dbu
  kind "in"   Ci = dBu at the input that reads 0 dBFS (loopback)             -> NAM output_level_dbu
The interface's own gain knobs (TotalMix, preamp gain, reference level) cannot be read, so an entry only holds while nobody touches them:
it keeps the date and a free note, and an input entry remembers the Co it was derived from (`co_at_time`).
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import fsutil

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "calibration.json"
STALE_CO_DB = 0.3            # an input calibrated against a Co that has since moved by more than this should be redone
_lock = threading.Lock()


def key_for(kind: str, device: str, channel: int, rate: int) -> str:
    return f"{kind}|{device}|{channel}|{rate}"


def _load() -> dict:
    try:
        data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data.get("entries"), dict):
        data = {"version": 1, "entries": {}}
    return data


def all_entries() -> dict:
    return _load()["entries"]


def get(kind: str, device: str, channel: int, rate: int) -> Optional[dict]:
    return _load()["entries"].get(key_for(kind, device, channel, rate))


def put(kind: str, device: str, channel: int, rate: int, **values) -> dict:
    """Store a calibration (replacing the previous one for the same channel) and return it."""
    entry = {"kind": kind, "device": device, "channel": int(channel), "rate": int(rate),
             "date": datetime.now().isoformat(timespec="milliseconds"), **values}
    with _lock:
        data = _load()
        data["entries"][key_for(kind, device, channel, rate)] = entry
        fsutil.write_text_atomic(CALIBRATION_FILE, json.dumps(data, indent=1))
    return entry


def delete(kind: str, device: str, channel: int, rate: int) -> bool:
    with _lock:
        data = _load()
        removed = data["entries"].pop(key_for(kind, device, channel, rate), None) is not None
        if removed:
            fsutil.write_text_atomic(CALIBRATION_FILE, json.dumps(data, indent=1))
    return removed


def is_stale(entry: dict, current_co: Optional[float]) -> bool:
    """An input entry whose reference output level has moved since."""
    if entry.get("kind") != "in" or current_co is None or entry.get("co_at_time") is None:
        return False
    return abs(current_co - entry["co_at_time"]) > STALE_CO_DB
