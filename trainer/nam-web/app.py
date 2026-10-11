"""Web GUI to train Neural Amp Modeler models on this machine (NVIDIA GPU, AMD GPU or CPU).

Upload the DI input and the reamped output(s) from this PC or, if the LAN switch is on, from any
device on the LAN; watch training progress, download the resulting .nam. Jobs run one at a time in
a separate worker process (worker.py). The page speaks Italian and English (i18n.py, static/i18n.json).

Everything the user cares about lives in two folders next to nam-web/, and the web
page is a live mirror of both:

  NAM Generati/<name>/      <name>.nam, <name>.png, <name>.log and _lavoro/ (job settings)
                            (any other .nam dropped anywhere in there is listed too)
  File WAV caricati/[sub/]  every uploaded WAV; files/subfolders added by hand show up as well

Nothing else is written outside nam-web/: uploads are spooled and temp files live in
nam-web/tmp, and the workers get TEMP/TMP/MPLCONFIGDIR pointed there too.
"""

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue
from typing import List, Optional

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import fsutil
import hardware
import i18n
import paths
import settings
import trainers
import updater
from i18n import msg, render, tr

HERE = Path(__file__).resolve().parent
LIB_DIR = paths.LIB_DIR          # all locations come from paths.py (NAM_DATA_DIR moves the data away from a read-only program folder)
WAV_DIR = paths.WAV_DIR
TMP_DIR = paths.TMP_DIR
CACHE_DIR = paths.CACHE_DIR
WORKER = HERE / "worker.py"
WORK = "_lavoro"  # per-model folder holding job settings, state and scratch files

GEAR_TYPES = ["amp", "pedal", "pedal_amp", "amp_cab", "amp_pedal_cab", "preamp", "studio"]
TONE_TYPES = ["clean", "overdrive", "crunch", "hi_gain", "fuzz"]

lock = threading.RLock()
jobs: dict = {}       # id -> job state (persisted in <model folder>/_lavoro/state.json)
procs: dict = {}
job_queue: "Queue[str]" = Queue()

paths.prepare()
tempfile.tempdir = str(TMP_DIR)  # multipart uploads spool here instead of the system temp


def jobs_active() -> bool:
    with lock:
        return any(j["state"] in ("queued", "running") for j in jobs.values())


updater.configure(jobs_active=jobs_active, tmp_dir=TMP_DIR, cache_dir=CACHE_DIR)


def err(status: int, key: str, **params) -> HTTPException:
    """An HTTP error whose text is rendered in the language of the request."""
    return HTTPException(status, tr(key, **params))


# -------------------------------------------------------------------- naming

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_name(name: str, fallback: str = "unnamed", max_len: int = 90) -> str:
    """A string that is a valid Windows file/folder name."""
    name = re.sub(r"\s+", " ", _INVALID_CHARS.sub("_", name)).strip(" .")[:max_len].strip(" .")
    if not name:
        return fallback
    return f"_{name}" if name.split(".")[0].upper() in _RESERVED else name


def safe_subfolder(value: str) -> Path:
    """Relative sub-path for uploads, e.g. 'Fender/Twin'. No traversal, no hidden/internal names."""
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
    raise err(500, "srv.too_many_files")


def make_unique_dir(parent: Path, base: str) -> Path:
    for i in range(1, 1000):
        candidate = parent / (base if i == 1 else f"{base} ({i})")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise err(500, "srv.too_many_folders")


# ------------------------------------------------------------------ WAV library

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


def capture_levels(wav_path: Path) -> Optional[dict]:
    """input_level_dbu/output_level_dbu from a NAM Capture sidecar .json next to this WAV (e.g. one saved
    under 'Da NAM Capture' by /nam-trainer/import), if there is one. None if there is no usable sidecar."""
    try:
        data = json.loads(wav_path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return None
    out = {k: metadata[k] for k in ("input_level_dbu", "output_level_dbu") if isinstance(metadata.get(k), (int, float))}
    return out or None


def list_wavs() -> dict:
    files, folders = [], []
    for dirpath, dirnames, filenames in os.walk(WAV_DIR):
        dirnames[:] = sorted((d for d in dirnames if not d.startswith((".", "_"))), key=str.lower)
        directory = Path(dirpath)
        rel_dir = directory.relative_to(WAV_DIR).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        if rel_dir:
            folders.append(rel_dir)
        for name in sorted(filenames, key=str.lower):
            if not name.lower().endswith(".wav"):
                continue
            path = directory / name
            info = wav_info(path)
            if info is None:
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            files.append({"rel": f"{rel_dir}/{name}" if rel_dir else name, "folder": rel_dir, "name": name,
                          "size": st.st_size, "mtime": st.st_mtime, "capture_levels": capture_levels(path), **info})
    return {"files": files, "folders": folders}


def wav_from_ref(ref: str) -> Path:
    """Resolve a path relative to 'File WAV caricati', refusing anything that escapes it."""
    path = (WAV_DIR / ref).resolve()
    if WAV_DIR.resolve() not in path.parents or path.suffix.lower() != ".wav" or not path.is_file():
        raise err(404, "srv.wav_not_found", ref=ref)
    return path


def wav_rel(path: Path) -> str:
    return path.resolve().relative_to(WAV_DIR.resolve()).as_posix()


def wav_rate(rel: str) -> int:
    info = wav_info(WAV_DIR / rel)
    if info is None:
        raise err(400, "srv.wav_unreadable", name=rel)
    return info["samplerate"]


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


def store_wav(upload: UploadFile, subfolder: str, created: List[Path]) -> str:
    """Move an uploaded WAV into 'File WAV caricati' (reusing an identical existing file)."""
    name = Path((upload.filename or "audio.wav").replace("\\", "/")).name
    stem = safe_name(Path(name).stem, fallback="audio")
    target_dir = WAV_DIR / safe_subfolder(subfolder)
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp = TMP_DIR / f"upload-{uuid.uuid4().hex}.wav"
    try:
        digest = save_upload(upload, tmp)
        if wav_info(tmp) is None:
            raise err(400, "srv.wav_unreadable", name=name)
        dest = target_dir / f"{stem}.wav"
        if dest.exists() and sha256_file(dest) == digest:
            return wav_rel(dest)
        dest = unique_file(dest)
        shutil.move(str(tmp), dest)
        created.append(dest)
        return wav_rel(dest)
    finally:
        tmp.unlink(missing_ok=True)
        _wav_cache.pop(str(tmp), None)


def store_capture_bytes(data: bytes, filename: str, subfolder: str) -> Path:
    """Like store_wav, but for bytes already in hand (NAM Capture sends the WAV as base64, not multipart)."""
    stem = safe_name(Path((filename or "audio.wav").replace("\\", "/")).stem, fallback="audio")
    target_dir = WAV_DIR / safe_subfolder(subfolder)
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp = TMP_DIR / f"capture-{uuid.uuid4().hex}.wav"
    try:
        tmp.write_bytes(data)
        if wav_info(tmp) is None:
            raise err(400, "srv.cap_bad_wav")
        dest = unique_file(target_dir / f"{stem}.wav")
        shutil.move(str(tmp), dest)
        return dest
    finally:
        tmp.unlink(missing_ok=True)
        _wav_cache.pop(str(tmp), None)


# ---------------------------------------------------------- model library / jobs


def persist(job: dict) -> None:
    path = job["_dir"] / WORK / "state.json"
    data = {k: v for k, v in job.items() if not k.startswith("_") and k != "cancel_requested"}
    fsutil.write_text_atomic(path, json.dumps(data))


def persist_safe(job: dict) -> None:
    try:
        persist(job)
    except OSError:
        pass


def scan_library():
    """Walk 'NAM Generati': (folders created by this app, loose .nam files found elsewhere)."""
    job_dirs, loose = [], []
    for dirpath, dirnames, filenames in os.walk(LIB_DIR):
        directory = Path(dirpath)
        if (directory / WORK / "state.json").is_file():
            job_dirs.append(directory)
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d != WORK and not d.startswith(".")]
        loose.extend(directory / n for n in sorted(filenames, key=str.lower) if n.lower().endswith(".nam"))
    return job_dirs, loose


def sync_jobs(startup: bool = False) -> List[Path]:
    """Make the in-memory jobs match the folders on disk. Returns the loose .nam files."""
    job_dirs, loose = scan_library()
    by_dir = {j["_dir"]: j for j in jobs.values()}
    found = set(job_dirs)
    fresh = []
    for directory in job_dirs:
        if directory in by_dir:
            continue
        try:
            state = json.loads((directory / WORK / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not state.get("id"):
            continue
        if state["id"] in jobs:  # the folder was moved/renamed by hand
            jobs[state["id"]]["_dir"] = directory
            continue
        state["_dir"] = directory
        if state.get("state") == "running" or (state.get("state") == "queued" and not startup):
            state.update(state="failed", finished=time.time(),
                         error=msg("job.interrupted") if startup else msg("job.imported_running"))
            jobs[state["id"]] = state
            # scratch files of a run that was killed half way through
            shutil.rmtree(directory / WORK / "train", ignore_errors=True)
            (directory / WORK / "stop").unlink(missing_ok=True)
            persist_safe(state)
        else:
            jobs[state["id"]] = state
            fresh.append(state)
    for job_id, job in list(jobs.items()):
        if job["_dir"] not in found and job["state"] != "running":
            del jobs[job_id]  # folder deleted by hand
    if startup:
        for job in sorted(fresh, key=lambda j: j["created"]):
            if job["state"] == "queued":
                job_queue.put(job["id"])
    return loose


def find_model(job: dict) -> Optional[Path]:
    path = job["_dir"] / f"{job['basename']}.nam"
    if path.exists():
        return path
    return next(iter(sorted(job["_dir"].glob("*.nam"))), None)


def rel_folder(directory: Path) -> str:
    rel = directory.relative_to(LIB_DIR).as_posix()
    return "" if rel == "." else rel


def public_job(job: dict) -> dict:
    out = {k: v for k, v in job.items() if not k.startswith("_") and k != "cancel_requested"}
    out["error"] = render(job.get("error")) or None  # stored language-neutral, shown in the reader's language
    directory, base = job["_dir"], job["basename"]
    progress_file = directory / WORK / "progress.json"
    if progress_file.exists():
        try:
            out["progress"] = json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass  # being replaced right now; the next poll will get it
    model = find_model(job)
    out.update(
        kind="job",
        folder=rel_folder(directory),
        has_model=model is not None,
        model_size=model.stat().st_size if model else None,
        has_plot=(directory / f"{base}.png").exists(),
        has_log=(directory / f"{base}.log").exists(),
        stopping=job["state"] == "running" and (directory / WORK / "stop").exists(),
        deletable=job["state"] != "running",
        wavs_ok=all((WAV_DIR / job.get(k, "")).is_file() for k in ("input_rel", "output_rel") if job.get(k)),
    )
    return out


def loose_id(path: Path) -> str:
    return "f-" + hashlib.sha1(path.relative_to(LIB_DIR).as_posix().lower().encode()).hexdigest()[:12]


def public_loose(path: Path) -> Optional[dict]:
    try:
        st = path.stat()
    except OSError:
        return None
    return {
        "id": loose_id(path), "kind": "file", "name": path.stem, "state": "done",
        "folder": rel_folder(path.parent), "created": st.st_mtime, "finished": st.st_mtime,
        "has_model": True, "model_size": st.st_size,
        "has_plot": path.with_suffix(".png").exists(), "has_log": path.with_suffix(".log").exists(),
        "deletable": False,
    }


def entry_files(entry_id: str) -> dict:
    """model/plot/log paths (+ display name) of a job or a loose .nam file."""
    with lock:
        loose = sync_jobs()
        job = jobs.get(entry_id)
        if job is not None:
            base, directory = job["basename"], job["_dir"]
            return {"name": base, "model": find_model(job), "plot": directory / f"{base}.png",
                    "log": directory / f"{base}.log"}
        for path in loose:
            if loose_id(path) == entry_id:
                return {"name": path.stem, "model": path, "plot": path.with_suffix(".png"),
                        "log": path.with_suffix(".log")}
    raise err(404, "srv.model_not_found")


def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise err(404, "srv.job_not_found")
    return job


# ------------------------------------------------------------------ scheduler


def run_job(job: dict) -> None:
    directory = job["_dir"]
    work = directory / WORK
    CACHE_DIR.mkdir(exist_ok=True)
    train_scratch = TMP_DIR / f"train-{job['id']}"  # short path on purpose: see worker.py
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "NAM_WEB_SERVER_PID": str(os.getpid()),
           "MPLBACKEND": "Agg", "TEMP": str(TMP_DIR), "TMP": str(TMP_DIR),
           "MPLCONFIGDIR": str(CACHE_DIR / "matplotlib"), "NAM_TRAIN_DIR": str(train_scratch)}
    # a training starts at HIGH priority (the worker also sets everything else: priority.py) - never REALTIME, which can starve the audio drivers
    flags = (subprocess.CREATE_NO_WINDOW | 0x00000080) if sys.platform == "win32" else 0
    with open(directory / f"{job['basename']}.log", "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, str(WORKER), str(directory), str(WAV_DIR)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=flags,
        )
        with lock:
            procs[job["id"]] = proc
        code = proc.wait()
    with lock:
        procs.pop(job["id"], None)
        result_file = work / "result.json"
        result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else None
        job["finished"] = time.time()
        if job.get("cancel_requested"):
            job["state"] = "cancelled"
        elif result and result.get("ok"):
            job.update(state="done", esr=result.get("esr"), stopped_early=result.get("stopped_early", False))
        else:
            error = (result or {}).get("error") or msg("job.exit_code", code=code)
            job.update(state="failed", error=error)
        shutil.rmtree(train_scratch, ignore_errors=True)  # checkpoints of a failed/cancelled/killed run
        shutil.rmtree(work / "train", ignore_errors=True)  # (older layout)
        for leftover in ("stop", "result.json"):
            (work / leftover).unlink(missing_ok=True)
        persist_safe(job)


def scheduler() -> None:
    while True:
        job_id = job_queue.get()
        while updater.is_exclusive():  # a trainer is being swapped: hold the job back until it is done
            time.sleep(1)
        with lock:
            job = jobs.get(job_id)
            if job is None or job["state"] != "queued":
                continue
            job.update(state="running", started=time.time())
            persist_safe(job)
        try:
            run_job(job)
        except Exception as e:  # keep the scheduler alive whatever happens
            with lock:
                job.update(state="failed", error=msg("job.error_raw", detail=f"{type(e).__name__}: {e}"), finished=time.time())
                persist_safe(job)


def exit_when_launcher_dies() -> None:
    """The launcher (tray / menu bar program) starts this server and passes its pid: if it is gone - killed, crashed - do not stay behind.
    Windows needs nothing (the launcher puts the server in a job object that dies with it); on macOS / Linux the pid is polled."""
    try:
        parent = int(os.environ.get("NAM_PARENT_PID", "0"))
    except ValueError:
        return
    if parent <= 0 or sys.platform == "win32":
        return

    def watch() -> None:
        while True:
            time.sleep(3)
            try:
                os.kill(parent, 0)
            except ProcessLookupError:
                os._exit(0)          # workers see their parent (this server) gone and stop by themselves (worker.exit_when_server_dies)
            except OSError:
                pass

    threading.Thread(target=watch, daemon=True).start()


def clean_tmp() -> None:
    for item in TMP_DIR.iterdir():
        try:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        except OSError:
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for directory in (LIB_DIR, WAV_DIR, TMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    clean_tmp()  # leftovers of an upload interrupted by a crash
    for note in updater.recover():  # a trainer swap interrupted by a crash / "Ferma NAM.bat"
        print(f"[aggiornamenti] {note}", flush=True)
    with lock:
        sync_jobs(startup=True)
    threading.Thread(target=scheduler, daemon=True).start()
    exit_when_launcher_dies()
    yield


app = FastAPI(title="NAM Trainer", lifespan=lifespan)

SERVER_PORT = 8765  # set from --port in __main__


@app.middleware("http")
async def language_and_access(request: Request, call_next):
    """1. The language of the reader (X-Lang header, ?lang=, or the browser's) for every text the server renders.
    2. The LAN switch: with LAN off, only this PC may use the page; anyone else gets a 403."""
    header = request.headers.get("x-lang") or request.query_params.get("lang") or request.headers.get("accept-language") or ""
    token = i18n.set_language(header)
    try:
        host = request.client.host if request.client else ""
        if not settings.lan_enabled() and not settings.is_local_client(host):
            text = tr("srv.lan_off")
            if request.url.path.startswith("/api/"):
                return JSONResponse({"detail": text}, status_code=403)
            return HTMLResponse(f"<!doctype html><meta charset=utf-8><title>NAM Trainer</title>"
                                f"<body style='font:16px system-ui;margin:3em auto;max-width:34em;padding:0 1em'>"
                                f"<h2>NAM Trainer</h2><p>{text}</p></body>", status_code=403)
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"      # the page files change with every update: always ask, the ETag keeps it cheap
        return response
    finally:
        i18n.reset_language(token)


# -------------------------------------------------------------------- helpers


def parse_optional(value: str, cast, label_key: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return cast(value)
    except ValueError:
        raise err(400, "srv.invalid_value", label=tr(label_key), value=value)


def reloaded_params(*, lr, lr_decay, lr_scheduler, fit_mrstft, stage_mode, stage2_epochs, stage2_lr,
                    stage2_lr_decay, stage2_lr_scheduler, stage2_focus) -> dict:
    """Validate the options that only the Reloaded trainers understand."""
    options = trainers.reloaded_options()
    for value, allowed, label_key in (
        (lr_scheduler, options["schedulers"], "field.scheduler"),
        (stage2_lr_scheduler, options["schedulers"], "field.scheduler2"),
        (stage_mode, options["stage_modes"], "field.stage_mode"),
        (stage2_focus, options["stage2_focus"], "field.stage2_focus"),
    ):
        if value and value not in allowed:
            raise err(400, "srv.invalid_value", label=tr(label_key), value=value)
    out = {
        "lr": parse_optional(lr, float, "field.lr"),
        "lr_decay": parse_optional(lr_decay, float, "field.lr_decay"),
        "lr_scheduler": lr_scheduler or None,
        "fit_mrstft": fit_mrstft,
        "stage_mode": stage_mode or "single_stage",
    }
    if out["stage_mode"] == "two_stage":
        out.update(
            stage2_epochs=parse_optional(stage2_epochs, int, "field.stage2_epochs"),
            stage2_lr=parse_optional(stage2_lr, float, "field.stage2_lr"),
            stage2_lr_decay=parse_optional(stage2_lr_decay, float, "field.stage2_lr_decay"),
            stage2_lr_scheduler=stage2_lr_scheduler or None,
            stage2_focus=stage2_focus or trainers.DEFAULTS["reloaded"]["stage2_focus"],
        )
        if not out["stage2_epochs"] or not 1 <= out["stage2_epochs"] <= 5000:
            raise err(400, "srv.stage2_epochs_range")
    for key, label_key in (("lr", "field.lr"), ("lr_decay", "field.lr_decay"), ("stage2_lr", "field.stage2_lr"),
                           ("stage2_lr_decay", "field.stage2_lr_decay")):
        if out.get(key) is not None and out[key] <= 0:
            raise err(400, "srv.must_be_positive", label=tr(label_key))
    return out


# ------------------------------------------------------------------------ API


@app.get("/api/config")
def config():
    return {
        "gear_types": GEAR_TYPES,
        "tone_types": TONE_TYPES,
        "trainers": trainers.catalog(),
        "defaults": trainers.defaults(),
        "reloaded": trainers.reloaded_options(),
        "hardware": {"variant": hardware.variant(), "cpu_name": hardware.cpu_name()},
        "default_language": hardware.language(),
        "paths": {"library": str(LIB_DIR), "wavs": str(WAV_DIR), "library_name": LIB_DIR.name,
                  "wavs_name": WAV_DIR.name},
    }


@app.get("/api/gpu")
def gpu():
    """The chip in the page header: live NVIDIA figures, or just the device name for AMD / CPU."""
    return hardware.gpu_info()


def settings_payload() -> dict:
    return {**settings.load(), "port": SERVER_PORT, "addresses": settings.lan_addresses(), "address": settings.primary_address()}


def is_local(request: Request) -> bool:
    return settings.is_local_client(request.client.host if request.client else "")


@app.get("/api/settings")
def get_settings(request: Request):
    # local_client: the page is open on the PC that runs the trainer (it may open that PC's folders); from the LAN those buttons are hidden
    return {**settings_payload(), "local_client": is_local(request)}


class SettingsBody(BaseModel):
    lan: Optional[bool] = None


@app.post("/api/settings")
def set_settings(body: SettingsBody):
    """Change a setting. `lan`: True = other PCs of the network may use the page, False = only this PC."""
    if body.lan is not None:
        settings.update(lan=body.lan)
    return settings_payload()


@app.get("/api/wavs")
def wavs():
    """Mirror of 'File WAV caricati' (subfolders included)."""
    return list_wavs()


@app.get("/api/models")
def list_models():
    """Mirror of 'NAM Generati': jobs created by this app plus any loose .nam file."""
    with lock:
        loose = sync_jobs()
        queued = sorted((j for j in jobs.values() if j["state"] == "queued"), key=lambda j: j["created"])
        positions = {j["id"]: i + 1 for i, j in enumerate(queued)}
        entries = [{**public_job(j), "queue_position": positions.get(j["id"])} for j in jobs.values()]
        entries += [e for e in map(public_loose, loose) if e]
    return sorted(entries, key=lambda e: e["created"], reverse=True)


@app.post("/api/models")
def create_models(
    output_files: Optional[List[UploadFile]] = File(None),
    output_refs: Optional[List[str]] = Form(None),
    input_file: Optional[UploadFile] = File(None),
    input_ref: str = Form(""),
    wav_folder: str = Form(""),
    trainer: str = Form(trainers.OFFICIAL_ID),
    epochs: int = Form(trainers.DEFAULTS["official"]["epochs"]),
    batch_size: int = Form(trainers.DEFAULTS["official"]["batch_size"]),
    latency: str = Form(""),
    threshold_esr: str = Form(""),
    ignore_checks: bool = Form(False),
    # Only used by the Reloaded trainers; blank = that trainer's own default
    lr: str = Form(""),
    lr_decay: str = Form(""),
    lr_scheduler: str = Form(""),
    fit_mrstft: bool = Form(True),
    stage_mode: str = Form(""),
    stage2_epochs: str = Form(""),
    stage2_lr: str = Form(""),
    stage2_lr_decay: str = Form(""),
    stage2_lr_scheduler: str = Form(""),
    stage2_focus: str = Form(""),
    name: str = Form(""),
    modeled_by: str = Form(""),
    gear_type: str = Form(""),
    gear_make: str = Form(""),
    gear_model: str = Form(""),
    tone_type: str = Form(""),
    input_level_dbu: str = Form(""),
    output_level_dbu: str = Form(""),
):
    if updater.is_exclusive():
        raise err(409, "srv.trainer_updating")
    if not 1 <= epochs <= 5000:
        raise err(400, "srv.epochs_range")
    if not 1 <= batch_size <= 512:
        raise err(400, "srv.batch_range")
    if gear_type and gear_type not in GEAR_TYPES:
        raise err(400, "srv.gear_invalid")
    if tone_type and tone_type not in TONE_TYPES:
        raise err(400, "srv.tone_invalid")
    resolved = trainers.resolve(trainer)
    if resolved is None:
        raise err(400, "srv.trainer_unavailable", trainer=trainer)
    backend, architecture, trainer_label = resolved
    params = {
        "trainer": trainer,
        "epochs": epochs,
        "batch_size": batch_size,
        "latency": parse_optional(latency, int, "field.latency"),
        "threshold_esr": parse_optional(threshold_esr, float, "field.threshold_esr"),
        "ignore_checks": ignore_checks,
    }
    if backend == "reloaded":
        params.update(reloaded_params(
            lr=lr, lr_decay=lr_decay, lr_scheduler=lr_scheduler, fit_mrstft=fit_mrstft, stage_mode=stage_mode,
            stage2_epochs=stage2_epochs, stage2_lr=stage2_lr, stage2_lr_decay=stage2_lr_decay,
            stage2_lr_scheduler=stage2_lr_scheduler, stage2_focus=stage2_focus,
        ))
    shared_metadata = {
        "modeled_by": modeled_by.strip() or None,
        "gear_type": gear_type or None,
        "gear_make": gear_make.strip() or None,
        "gear_model": gear_model.strip() or None,
        "tone_type": tone_type or None,
        "input_level_dbu": parse_optional(input_level_dbu, float, "field.input_level"),
        "output_level_dbu": parse_optional(output_level_dbu, float, "field.output_level"),
    }
    short_label = trainer_label.split(" - ", 1)[-1]

    created_files: List[Path] = []
    created_dirs: List[Path] = []
    new_jobs = []
    try:
        if input_file is not None and input_file.filename:
            input_rel = store_wav(input_file, wav_folder, created_files)
        elif input_ref:
            input_rel = wav_rel(wav_from_ref(input_ref))
        else:
            raise err(400, "srv.choose_input")
        output_rels: List[str] = [wav_rel(wav_from_ref(ref)) for ref in output_refs or []]
        for upload in output_files or []:
            if upload.filename:
                output_rels.append(store_wav(upload, wav_folder, created_files))
        output_rels = list(dict.fromkeys(output_rels))  # same file picked twice
        if not output_rels:
            raise err(400, "srv.add_output")

        in_rate = wav_rate(input_rel)
        for rel in output_rels:
            out_rate = wav_rate(rel)
            if out_rate != in_rate:
                raise err(400, "srv.sample_rate_mismatch", input_rate=in_rate, output=rel, output_rate=out_rate)

        for i, output_rel in enumerate(output_rels):
            stem = Path(output_rel).stem
            wanted = name.strip() if (name.strip() and len(output_rels) == 1) else f"{stem} ({short_label})"
            directory = make_unique_dir(LIB_DIR, safe_name(wanted))
            created_dirs.append(directory)
            (directory / WORK).mkdir()
            basename = directory.name
            (directory / WORK / "job.json").write_text(json.dumps({
                "backend": backend, "architecture": architecture, "basename": basename,
                "input_rel": input_rel, "output_rel": output_rel, **params,
                "metadata": {**shared_metadata, "name": basename},
            }), encoding="utf-8")
            new_jobs.append({
                "id": uuid.uuid4().hex[:12],
                "name": basename,
                "basename": basename,
                "state": "queued",
                "created": time.time() + i * 1e-3,  # keeps batch order stable
                "started": None,
                "finished": None,
                "error": None,
                "esr": None,
                "input_rel": input_rel,
                "output_rel": output_rel,
                "input_name": Path(input_rel).name,
                "output_name": Path(output_rel).name,
                "trainer_label": trainer_label,
                "trainer_version": trainers.trainer_version(backend),  # which release produced this model
                "params": params,
                "metadata": shared_metadata,
                "_dir": directory,
            })
    except BaseException:
        for directory in created_dirs:
            shutil.rmtree(directory, ignore_errors=True)
        for path in created_files:
            path.unlink(missing_ok=True)
        raise

    with lock:
        for job in new_jobs:
            jobs[job["id"]] = job
            persist(job)
            job_queue.put(job["id"])
    return {"models": [j["id"] for j in new_jobs], "input_rel": input_rel}


@app.get("/api/models/{model_id}/log")
def model_log(model_id: str, lines: int = 300):
    log = entry_files(model_id)["log"]
    if not log.exists():
        return {"lines": []}
    with open(log, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 128 * 1024))
        text = f.read().decode("utf-8", errors="replace")
    # tqdm redraws with \r (and Windows ends lines with \r\n): keep only the latest state of each line
    out = [line.rstrip("\r").rsplit("\r", 1)[-1].rstrip() for line in text.split("\n")]
    return {"lines": [l for l in out if l][-max(1, min(lines, 2000)):]}


@app.post("/api/models/{model_id}/stop")
def model_stop(model_id: str):
    """Stop training early but still export the best model so far."""
    with lock:
        job = get_job(model_id)
        if job["state"] != "running":
            raise err(409, "srv.job_not_running")
        (job["_dir"] / WORK / "stop").touch()
    return {"ok": True}


@app.post("/api/models/{model_id}/cancel")
def model_cancel(model_id: str):
    with lock:
        job = get_job(model_id)
        if job["state"] == "queued":
            job.update(state="cancelled", finished=time.time())
            persist_safe(job)
        elif job["state"] == "running":
            job["cancel_requested"] = True
            proc = procs.get(model_id)
            if proc:
                proc.kill()
        else:
            raise err(409, "srv.job_finished")
    return {"ok": True}


@app.delete("/api/models/{model_id}")
def model_delete(model_id: str):
    """Delete a model folder created by this app (never loose files or anything outside 'NAM Generati')."""
    with lock:
        job = get_job(model_id)
        if job["state"] == "running":
            raise err(409, "srv.cancel_first")
        directory = job["_dir"].resolve()
        if LIB_DIR.resolve() not in directory.parents or not (directory / WORK / "state.json").is_file():
            raise err(409, "srv.not_deletable")
        del jobs[model_id]
    shutil.rmtree(directory, ignore_errors=True)
    return {"ok": True}


@app.get("/api/models/{model_id}/model")
def model_download(model_id: str):
    files = entry_files(model_id)
    if files["model"] is None or not files["model"].exists():
        raise err(404, "srv.model_unavailable")
    return FileResponse(files["model"], filename=f"{files['name']}.nam", media_type="application/octet-stream")


@app.get("/api/models/{model_id}/plot")
def model_plot(model_id: str):
    plot = entry_files(model_id)["plot"]
    if not plot.exists():
        raise err(404, "srv.plot_unavailable")
    return FileResponse(plot, media_type="image/png")


class InstallBody(BaseModel):
    target: str


class RestoreBody(BaseModel):
    backup: str


def _run_update_call(name: str, fn, *args) -> dict:
    if name not in updater.SPECS:
        raise err(404, "srv.unknown_trainer")
    try:
        fn(name, *args)
    except updater.UpdateError as e:  # includes Busy
        raise HTTPException(409, tr(e.key, **e.params))
    return {"ok": True}


@app.get("/api/updates")
def updates_status():
    """State of the two trainers, last check/operation results, backups and the live log."""
    return updater.status(i18n.get_language())


@app.post("/api/updates/{name}/check")
def updates_check(name: str):
    return _run_update_call(name, updater.start_check)


@app.post("/api/updates/{name}/install")
def updates_install(name: str, body: InstallBody):
    return _run_update_call(name, updater.start_install, body.target)


@app.post("/api/updates/{name}/restore")
def updates_restore(name: str, body: RestoreBody):
    return _run_update_call(name, updater.start_restore, body.backup)


CAPTURE_SUBFOLDER = "Da NAM Capture"


@app.get("/nam-trainer/ping")
def nam_trainer_ping():
    """NAM Capture calls this to know whether a trainer is listening at an address before offering it."""
    return {"ok": True}


INPUT_SUBFOLDER = "Input"      # the input.wav files NAM Capture played to the amp: kept apart from the results, like in NAM Capture


class CaptureImportBody(BaseModel):
    name: str = ""
    input_rel: str = ""
    input_md5: str = ""        # the input.wav played during the capture: asked for (see /nam-trainer/import-input) only if not here yet
    input_size: int = 0
    metadata: dict = {}
    report: dict = {}
    wav_name: str = "cattura.wav"
    wav_base64: str


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def find_library_wav(md5: str, size: int) -> Optional[str]:
    """The library path of a WAV with exactly this content, if there is one. The size is compared first: only a file of
    the same size is read to be hashed."""
    if not re.fullmatch(r"[0-9a-f]{32}", md5) or size <= 0:
        return None
    for dirpath, dirnames, filenames in os.walk(WAV_DIR):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "_"))]
        for name in filenames:
            path = Path(dirpath) / name
            if not name.lower().endswith(".wav"):
                continue
            try:
                if path.stat().st_size == size and file_md5(path) == md5:
                    return wav_rel(path)
            except OSError:
                continue
    return None


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@app.post("/nam-trainer/import")
def nam_trainer_import(body: CaptureImportBody):
    """Receive a finished capture from NAM Capture: store the WAV + a sidecar .json with its calibration
    metadata and quality report, so it shows up in 'File WAV caricati' for 'Crea modello' like any other file.
    Does not create a model by itself. The answer says whether the input.wav of the capture is already in the library
    (`input_rel`) or has to be sent too (`input_needed`: NAM Capture then calls /nam-trainer/import-input)."""
    try:
        data = base64.b64decode(body.wav_base64, validate=True)
    except (binascii.Error, ValueError):
        raise err(400, "srv.cap_bad_base64")
    if not data:
        raise err(400, "srv.cap_bad_base64")
    dest = store_capture_bytes(data, body.wav_name or body.name, CAPTURE_SUBFOLDER)
    input_rel = (find_library_wav(body.input_md5, body.input_size) or "") if body.input_md5 else body.input_rel
    sidecar = {"name": body.name, "input_rel": input_rel, "input_md5": body.input_md5, "metadata": body.metadata, "report": body.report}
    dest.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"received": wav_rel(dest), "input_rel": input_rel, "input_needed": bool(body.input_md5 and not input_rel)}


class CaptureInputBody(BaseModel):
    capture_rel: str           # the capture this input belongs to, as answered by /nam-trainer/import ("received")
    name: str = "input.wav"
    md5: str
    wav_base64: str


@app.post("/nam-trainer/import-input")
def nam_trainer_import_input(body: CaptureInputBody):
    """The input.wav a capture was made with: stored in 'Input' (or recognised as a file the library already has) and written into
    the capture's sidecar, so the new-training form finds the right input by itself."""
    capture = wav_from_ref(body.capture_rel)
    try:
        data = base64.b64decode(body.wav_base64, validate=True)
    except (binascii.Error, ValueError):
        raise err(400, "srv.cap_bad_base64")
    if not data or hashlib.md5(data).hexdigest() != body.md5:
        raise err(400, "srv.cap_bad_input")
    input_rel = find_library_wav(body.md5, len(data))
    if input_rel is None:
        input_rel = wav_rel(store_capture_bytes(data, body.name, INPUT_SUBFOLDER))
    sidecar_path = capture.with_suffix(".json")
    sidecar = read_json(sidecar_path)
    sidecar["input_rel"] = input_rel
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"input_rel": input_rel}


@app.get("/api/captures")
def api_captures():
    """The WAV files that came from NAM Capture plus every other WAV a training has used as its output, newest first, each with what
    happened to it: no training yet ("new"), or the trainings made from it (newest first)."""
    with lock:
        sync_jobs()
        runs_of = {}
        for job in jobs.values():
            if job.get("output_rel"):
                runs_of.setdefault(job["output_rel"], []).append(job)
    rels = set(runs_of)
    received = WAV_DIR / CAPTURE_SUBFOLDER
    if received.is_dir():
        rels.update(wav_rel(p) for p in received.rglob("*.wav"))
    items = []
    for rel in rels:
        path = WAV_DIR / rel
        try:
            st = path.stat()
        except OSError:
            continue                          # deleted by hand, a training still names it
        sidecar = read_json(path.with_suffix(".json"))
        report = sidecar.get("report") if isinstance(sidecar.get("report"), dict) else {}
        runs = sorted(runs_of.get(rel, []), key=lambda j: j["created"], reverse=True)
        items.append({
            "rel": rel, "name": sidecar.get("name") or path.stem, "received": st.st_mtime, "size": st.st_size,
            "from_capture": rel.startswith(CAPTURE_SUBFOLDER + "/"),
            "levels": capture_levels(path), "input_rel": sidecar.get("input_rel") or "",
            "esr": report.get("replicate_esr"), "ok": report.get("ok"), "warnings": report.get("warnings") or [],
            "runs": [{"id": j["id"], "name": j["name"], "state": j["state"], "esr": j.get("esr"), "created": j["created"]} for j in runs],
        })
    items.sort(key=lambda i: i["received"], reverse=True)
    return items


@app.post("/api/captures/reveal")
def captures_reveal(request: Request):
    """"Sfoglia": opens the folder the captures arrive in, in the file manager of the PC that runs the trainer - only for a page open on that
    same PC (from another one it would pop a window up on somebody else's screen)."""
    if not is_local(request):
        raise err(403, "srv.local_only")
    folder = WAV_DIR / CAPTURE_SUBFOLDER
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(str(folder))                                    # noqa: S606 - our own data folder
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])
    return {"path": str(folder)}


@app.delete("/api/captures")
def capture_delete(rel: str):
    """Remove one WAV of the received / processed list (and the capture's small .json next to it). Refused while a training
    that is queued or running still uses it; the models already made from it stay."""
    path = wav_from_ref(rel)
    with lock:
        sync_jobs()
        if any(j.get("output_rel") == rel and j["state"] in ("queued", "running") for j in jobs.values()):
            raise err(409, "srv.cap_in_use")
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"ok": True}


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")


# ----------------------------------------------------------------------- main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # Always listen on every interface: the LAN switch (settings.json) is enforced per request, so it can be
    # flipped from the page at any time without restarting the server.
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    SERVER_PORT = args.port
    lan = settings.lan_enabled()
    print(f"\n  NAM Trainer - {hardware.variant().upper()} - http://localhost:{args.port}")
    if lan:
        for ip in settings.lan_addresses():
            print(f"                       http://{ip}:{args.port}   (LAN)")
    else:
        print("                       LAN off: only this PC can use the page")
    print(f"\n  {LIB_DIR}\n  {WAV_DIR}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
