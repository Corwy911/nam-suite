"""Persistent user settings (nam-web/settings.json). Today: the LAN switch.

lan = True   the web page is reachable from other PCs of the local network (the default, as before)
lan = False  only this PC can use it, like a local app: requests from any other address get a 403
"""

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import List

import fsutil
import paths

SETTINGS_FILE = paths.SETTINGS_FILE
DEFAULTS = {"lan": True}

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


def primary_address() -> str:
    """The one address other PCs should type. A VPN adapter often owns the default route and an address like 10.x, so the
    route does not decide first: the 'home network' ranges win (192.168 before 172.16-31 before 10 before the rest) and the
    address the PC routes out with only breaks a tie. Empty when there is none."""
    found = lan_addresses()
    if not found:
        return ""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))                  # documentation address: only picks the outgoing interface, nothing is sent
            routed = probe.getsockname()[0]
    except OSError:
        routed = ""

    def rank(a: str) -> int:
        if a.startswith("192.168."):
            return 0
        if a.startswith("172.") and 16 <= int(a.split(".")[1]) <= 31:
            return 1
        return 2 if a.startswith("10.") else 3
    return min(found, key=lambda a: (rank(a), a != routed, a))


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
