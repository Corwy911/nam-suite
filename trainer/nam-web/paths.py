"""Where everything lives. Standard library only: every other module of nam-web (and worker.py) imports this one.

Default (no environment variable): exactly the historical layout, next to the program -
    <program>\\nam-web        the code, settings.json, hardware.json, tmp\\, cache\\
    <program>\\trainers       the two NAM trainers (the page can update them)
    <program>\\NAM Generati   models        <program>\\File WAV caricati   WAV files        <program>\\Backup   trainer backups

NAM_DATA_DIR=<folder>: the program folder is read-only (a macOS .app bundle, Program Files...), so everything that changes goes to <folder>:
    <folder>\\NAM Generati, File WAV caricati, Backup, trainers (copied from the program on first run), state\\ (settings.json, hardware.json, tmp, cache)
A hardware.json written next to the code by an installer is still found (the folder's own one wins).
"""

import os
import shutil
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parent                 # nam-web
PROGRAM_ROOT = PROGRAM_DIR.parent
_data = os.environ.get("NAM_DATA_DIR", "").strip()

CUSTOM = bool(_data)
DATA_DIR = Path(_data).expanduser() if CUSTOM else PROGRAM_ROOT
STATE_DIR = DATA_DIR / "state" if CUSTOM else PROGRAM_DIR
LIB_DIR = DATA_DIR / "NAM Generati"
WAV_DIR = DATA_DIR / "File WAV caricati"
BACKUP_DIR = DATA_DIR / "Backup"
TMP_DIR = STATE_DIR / "tmp"
CACHE_DIR = STATE_DIR / "cache"
SETTINGS_FILE = STATE_DIR / "settings.json"
TRAINERS_DIR = DATA_DIR / "trainers" if CUSTOM else PROGRAM_ROOT / "trainers"
BUNDLED_TRAINERS_DIR = PROGRAM_ROOT / "trainers"              # what ships with the program


def hardware_file() -> Path:
    for candidate in (STATE_DIR / "hardware.json", PROGRAM_DIR / "hardware.json"):
        if candidate.exists():
            return candidate
    return STATE_DIR / "hardware.json"


def prepare() -> None:
    """Create the data folders; with a custom data folder, bring in the bundled trainers the first time (never overwrite: the page may have updated them)."""
    for directory in (DATA_DIR, STATE_DIR, LIB_DIR, WAV_DIR, TMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if CUSTOM and BUNDLED_TRAINERS_DIR.is_dir():
        TRAINERS_DIR.mkdir(parents=True, exist_ok=True)
        for sub in BUNDLED_TRAINERS_DIR.iterdir():
            if sub.is_dir() and not (TRAINERS_DIR / sub.name).exists():
                shutil.copytree(sub, TRAINERS_DIR / sub.name)
