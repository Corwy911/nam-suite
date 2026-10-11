"""NAM Capture - standalone calibrated reamp capture tool. No trainer code here at all: this project only records and
calibrates; training happens on the Windows PC, reached later over the LAN (not built yet, see the manual, chapter 11).

Everything the user cares about lives in one folder next to this one, and the page is a live mirror of it:

  File WAV caricati/Catture/    every capture (WAV + a .json of its calibrated levels and report)

Nothing else is written outside this project: the capture engine's scratch files live in tmp/.
Normally started by the packaged app (see packaging/); can also be run directly for development:

    venv/bin/python app.py --port 8766
"""

import argparse
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import capture_api
import i18n
import wavlib
from capture import store as cal_store
from i18n import tr
from wavlib import safe_name, safe_subfolder, save_upload, unique_file

HERE = Path(__file__).resolve().parent

# Per-user data lives in the OS's standard per-user data location, not next to the program: a real app's files survive
# a rebuild of the app bundle and do not depend on where the program itself happens to be installed.
if sys.platform == "win32":
    APP_SUPPORT = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "NAM Capture"
else:
    APP_SUPPORT = Path.home() / "Library" / "Application Support" / "NAM Capture"
WAV_DIR = APP_SUPPORT / "File WAV caricati"
TMP_DIR = APP_SUPPORT / "tmp"
MODEL_DIR = APP_SUPPORT / "Modelli NAM"    # .nam profiles for the "Prova dal vivo" tool (capture_api/live.py)
IR_DIR = APP_SUPPORT / "IR"                # cabinet IRs (.wav), same tool
LIVE_USAGE_FILE = APP_SUPPORT / "live_usage.json"   # how many times each saved model/IR has been used, for "most used"
cal_store.CALIBRATION_FILE = APP_SUPPORT / "calibration.json"

for _d in (WAV_DIR, TMP_DIR, MODEL_DIR, IR_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# Thin wrappers that read the WAV_DIR/TMP_DIR *module globals* on every call (not captured once at import time), so that
# reassigning them - as the tests do, to point at a throwaway directory - is picked up without re-wiring anything.
def wav_rel(path: Path) -> str:
    return wavlib.wav_rel(WAV_DIR, path)


def wav_from_ref(ref: str) -> Path:
    return wavlib.wav_from_ref(WAV_DIR, ref)


def store_wav(upload, subfolder: str, created) -> str:
    return wavlib.store_wav(WAV_DIR, TMP_DIR, upload, subfolder, created)


capture_api.configure(wav_dir=WAV_DIR, tmp_dir=TMP_DIR, model_dir=MODEL_DIR, ir_dir=IR_DIR, usage_file=LIVE_USAGE_FILE,
                      safe_name=safe_name, safe_subfolder=safe_subfolder, unique_file=unique_file,
                      wav_rel=wav_rel, wav_from_ref=wav_from_ref, store_wav=store_wav, save_upload=save_upload)


def clean_tmp() -> None:
    """Leftovers of a run interrupted by a crash. Not live_uploads/ (capture_api/live.py's "Prova dal vivo" file
    picker): those are meant to survive a server restart between browsing a file and pressing "Avvia" - a
    capture-tool restart (or opening/reopening the packaged app for the native engine, see the two
    ".command" scripts) must not silently delete a file Roman just picked. Those copies are kept while
    they may still be needed (a second Avvia, a later swap), so they get a light day-old sweep here instead of a blanket wipe."""
    for item in TMP_DIR.iterdir():
        if item.name == "live_uploads":
            continue
        try:
            import shutil
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        except OSError:
            pass
    live_uploads = TMP_DIR / "live_uploads"
    if live_uploads.is_dir():
        cutoff = time.time() - 24 * 3600
        for item in live_uploads.iterdir():
            try:
                if item.stat().st_mtime < cutoff:
                    item.unlink()
            except OSError:
                pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    clean_tmp()  # leftovers of a run interrupted by a crash
    try:
        moved = capture_api.organize_library()
        if moved["inputs"] or moved["results"]:
            print(f"library tidied: inputs -> Input/ {moved['inputs']}, results -> own folders {moved['results']}", flush=True)
    except Exception as e:  # tidying is a courtesy: never keep the app from starting
        print(f"library tidy-up skipped: {e}", flush=True)
    yield
    capture_api.shutdown()  # release the audio interface (the engine process also leaves on its own if this dies)


app = FastAPI(title="NAM Capture", lifespan=lifespan)


@app.middleware("http")
async def language(request: Request, call_next):
    """The language of the reader (X-Lang header, ?lang=, or the browser's) for every text the server renders."""
    header = request.headers.get("x-lang") or request.query_params.get("lang") or request.headers.get("accept-language") or ""
    token = i18n.set_language(header)
    try:
        return await call_next(request)
    finally:
        i18n.reset_language(token)


app.include_router(capture_api.router)


@app.get("/health", include_in_schema=False)
def health():
    """Polled by the app shell while it waits for the server to come up."""
    return {"ok": True}


@app.get("/capture", include_in_schema=False)
def capture_alias():
    """The capture page is the whole app now; keep the old /capture URL working too."""
    return FileResponse(HERE / "static" / "capture.html", headers={"Cache-Control": "no-cache"})


class RevalidatingStatic(StaticFiles):
    """Every page file is revalidated (ETag) on each load: the app window's WebKit otherwise keeps serving a days-old app.js
    from its cache after the app is updated, so new page behaviour silently does not arrive."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", RevalidatingStatic(directory=HERE / "static", html=True), name="static")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")  # standalone tool, no LAN: only this Mac talks to it
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    print(f"\n  NAM Capture - http://{args.host}:{args.port}\n  {WAV_DIR}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
