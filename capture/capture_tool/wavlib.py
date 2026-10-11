"""The small WAV library the capture tool writes into ("File WAV caricati/Catture/…"): safe names, no path traversal, no
overwriting a different file by accident. Extracted from the original nam-trainer-gui app.py (same code, no behaviour
change) so the capture tool has no dependency on the trainer at all. Standard library + soundfile only.

Every function that touches the library takes `wav_dir` (and `tmp_dir`) explicitly rather than closing over it, so the
caller's current directory is always the one used - app.py's tests swap WAV_DIR/TMP_DIR at runtime and rely on that.
"""

import hashlib
import re
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

import soundfile as sf
from fastapi import HTTPException, UploadFile

from i18n import tr

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_name(name: str, fallback: str = "unnamed", max_len: int = 90) -> str:
    """A string that is a valid Windows file/folder name too (the capture may end up copied to the Windows trainer PC)."""
    name = re.sub(r"\s+", " ", _INVALID_CHARS.sub("_", name)).strip(" .")[:max_len].strip(" .")
    if not name:
        return fallback
    return f"_{name}" if name.split(".")[0].upper() in _RESERVED else name


def safe_subfolder(value: str) -> Path:
    """Relative sub-path for the library, e.g. 'Catture/King of Tone'. No traversal, no hidden/internal names."""
    parts = []
    for part in re.split(r"[\\/]+", value or ""):
        part = part.strip()
        if part in ("", ".", ".."):
            continue
        part = safe_name(part).lstrip("._") or "folder"
        parts.append(part)
    return Path(*parts[:6]) if parts else Path()


def unique_file(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 1000):
        candidate = path.with_name(f"{path.stem} ({i}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise HTTPException(500, tr("srv.cap.too_many_files"))


_wav_cache: dict = {}


def wav_info(path: Path) -> Optional[dict]:
    """sample rate / duration of a WAV, cached by mtime+size. None if unreadable."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    hit = _wav_cache.get(str(path))
    if hit and hit[0] == key:
        return hit[1]
    try:
        info = sf.info(str(path))
        value = {"samplerate": info.samplerate, "duration": round(info.duration, 1)}
    except Exception:
        value = None
    _wav_cache[str(path)] = (key, value)
    return value


def wav_from_ref(wav_dir: Path, ref: str) -> Path:
    """Resolve a path relative to the library, refusing anything that escapes it."""
    path = (wav_dir / ref).resolve()
    if wav_dir.resolve() not in path.parents or path.suffix.lower() != ".wav" or not path.is_file():
        raise HTTPException(404, tr("srv.cap.wav_not_found", ref=ref))
    return path


def wav_rel(wav_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(wav_dir.resolve()).as_posix()


def save_upload(upload: UploadFile, dest: Path) -> str:
    """Stream an upload to disk, returning its sha256."""
    digest = hashlib.sha256()
    with open(dest, "wb") as f:
        while chunk := upload.file.read(1 << 20):
            digest.update(chunk)
            f.write(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def store_wav(wav_dir: Path, tmp_dir: Path, upload: UploadFile, subfolder: str, created: List[Path]) -> str:
    """Move an uploaded WAV into the library (reusing an identical existing file)."""
    name = Path((upload.filename or "audio.wav").replace("\\", "/")).name
    stem = safe_name(Path(name).stem, fallback="audio")
    target_dir = wav_dir / safe_subfolder(subfolder)
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"upload-{uuid.uuid4().hex}.wav"
    try:
        digest = save_upload(upload, tmp)
        if wav_info(tmp) is None:
            raise HTTPException(400, tr("srv.cap.wav_unreadable", name=name))
        dest = target_dir / f"{stem}.wav"
        if dest.exists() and sha256_file(dest) == digest:
            return wav_rel(wav_dir, dest)
        dest = unique_file(dest)
        shutil.move(str(tmp), dest)
        created.append(dest)
        return wav_rel(wav_dir, dest)
    finally:
        tmp.unlink(missing_ok=True)
        _wav_cache.pop(str(tmp), None)
