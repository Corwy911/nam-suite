"""Persistent user settings (nam-web/settings.json). Today: the LAN switch and the trainer's address.

lan = True             the web page is reachable from other PCs of the local network (the default, as before)
lan = False             only this PC can use it, like a local app: requests from any other address get a 403
trainer_host = ""       the hostname/IP of the PC running nam-trainer-gui, probed live on the LAN (see
                        capture_api/trainer.py) - empty means "not set up yet"
"""

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import List

import fsutil

SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"
DEFAULTS = {"lan": True, "trainer_host": ""}

_lock = threading.Lock()


def load() -> dict:
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {**DEFAULTS, **{k: data[k] for k in DEFAULTS if k in data and isinstance(data[k], type(DEFAULTS[k]))}}


def update(**changes) -> dict:
    with _lock:
        current = load()
        for key, value in changes.items():
            if key in DEFAULTS and isinstance(value, type(DEFAULTS[key])):
                current[key] = value
        fsutil.write_text_atomic(SETTINGS_FILE, json.dumps(current, indent=1))
        return current


def lan_enabled() -> bool:
    return bool(load()["lan"])


# ------------------------------------------------------------- who is "this PC"?

_LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}
_local_cache = {"t": 0.0, "addresses": set()}


def lan_addresses() -> List[str]:
    """IPv4 addresses of this PC on the network (without loopback / link-local)."""
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    return sorted({i[4][0] for i in infos if not i[4][0].startswith(("127.", "169.254."))})


def _own_addresses() -> set:
    now = time.time()
    if now - _local_cache["t"] > 30:
        found = set(lan_addresses())
        try:
            found.update(a for a in socket.gethostbyname_ex(socket.gethostname())[2])
        except OSError:
            pass
        _local_cache.update(t=now, addresses=found)
    return _local_cache["addresses"]


def is_local_client(host: str) -> bool:
    """True if the request comes from this very PC: loopback, or one of the PC's own addresses
    (someone opening http://<this PC's IP>:8765 on this PC arrives with that IP as source)."""
    if not host:
        return False
    host = host.strip().lower()
    if host in _LOOPBACK or host.startswith("::ffff:127."):
        return True
    return host.removeprefix("::ffff:") in _own_addresses()
