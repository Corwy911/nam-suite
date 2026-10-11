"""Small file helpers shared by the server, the worker and the settings (standard library only).

On Windows, os.replace() fails with "Accesso negato" (WinError 5) or a sharing violation (WinError 32) if another
program has the target open at that very moment: the page polling the server every 2 s, an antivirus, the indexer.
A live progress file that is written several times per second and read by the server is exactly where that happens,
so every "write to .tmp, then replace" goes through replace_retry().
"""

import os
import time
from pathlib import Path


def replace_retry(tmp: Path, path: Path, seconds: float = 3.0) -> None:
    """os.replace(tmp, path), retrying for a short while when the target is momentarily locked."""
    deadline = time.monotonic() + seconds
    delay = 0.01
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.25)


def write_text_atomic(path: Path, text: str, seconds: float = 3.0) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    replace_retry(tmp, path, seconds)
