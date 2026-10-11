"""Which accelerator this installation was set up for (hardware.json, written by the installer; see paths.py for where it lives).

  variant "nvidia"  CUDA build of PyTorch, NVIDIA GPU
  variant "amd"     ROCm build of PyTorch (AMD's Windows preview), Radeon GPU
  variant "mps"     PyTorch on the Apple GPU (Metal Performance Shaders), Apple Silicon Macs
  variant "cpu"     CPU build of PyTorch: works everywhere, but training is much slower

No file (older installs, development copy) = "mps" on an Apple Silicon Mac, "nvidia" if nvidia-smi exists, otherwise "cpu".
Only the standard library is used here: worker.py imports this module too.
"""

import json
import paths
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

VARIANTS = ("nvidia", "amd", "mps", "cpu")
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _nvidia_smi() -> Optional[str]:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    candidate = Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    return str(candidate) if candidate.exists() else None


def _apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _apple_chip() -> str:
    """"Apple M1 Max" and the like (sysctl; the chip is also the GPU)."""
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
        return out or "Apple Silicon"
    except (OSError, subprocess.SubprocessError):
        return "Apple Silicon"


def load() -> dict:
    try:
        data = json.loads(paths.hardware_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if data.get("variant") not in VARIANTS:
        guess = "mps" if _apple_silicon() else "nvidia" if _nvidia_smi() else "cpu"
        data = {**data, "variant": guess, "assumed": True}
    return data


def variant() -> str:
    return load()["variant"]


def language() -> str:
    """The language chosen when installing ("it" | "en"); "" for a development copy. The page uses it as its default."""
    value = load().get("language")
    return value if value in ("it", "en") else ""


def cpu_name() -> str:
    return load().get("cpu_name") or (_apple_chip() if sys.platform == "darwin" else "") or platform.processor() or "CPU"


_gpu_cache = {"t": 0.0, "value": None}


def gpu_info() -> Optional[dict]:
    """What the header chip shows: live NVIDIA figures, or just the device name for AMD / Apple / CPU."""
    hw = load()
    kind = hw["variant"]
    if kind == "amd":
        return {"kind": "amd", "name": hw.get("gpu_name") or "AMD Radeon"}
    if kind == "mps":
        return {"kind": "mps", "name": hw.get("gpu_name") or _apple_chip()}
    if kind == "cpu":
        return {"kind": "cpu", "name": cpu_name()}
    if time.time() - _gpu_cache["t"] < 2:
        return _gpu_cache["value"]
    value = None
    smi = _nvidia_smi()
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True, creationflags=_NO_WINDOW,
            ).stdout.strip().splitlines()[0]
            name, util, used, total, temp = [x.strip() for x in out.split(",")]
            value = {"kind": "nvidia", "name": name, "util": int(util), "mem_used": int(used), "mem_total": int(total), "temp": int(temp)}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    _gpu_cache.update(t=time.time(), value=value)
    return value
