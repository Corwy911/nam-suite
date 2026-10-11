"""Shared plumbing of the capture API: the folder/helper context app.py passes in, the engine session, error helpers, the
native-engine relay and the small shared state object. No routes here."""

import json
import os
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException

from capture import alignment as al
from capture import store
from capture.session import EngineError, EngineSession
from i18n import tr

RATE = 48_000
CAPTURE_FOLDER = "Catture"          # under "File WAV caricati": one sub-folder per capture (see _new_take_dir)
INPUT_FOLDER = "Input"              # same library, own folder: the input.wav files played to the amp are not results
EXTRA_TAIL = 48_000                 # recorded after the input ends, so the delayed end of the response is not lost

_ctx: dict = {}
_session: Optional[EngineSession] = None
_lock = threading.RLock()
_input_cache: dict = {}


def configure(*, wav_dir: Path, tmp_dir: Path, model_dir: Path, ir_dir: Path, usage_file: Path, safe_name,
              safe_subfolder, unique_file, wav_rel, wav_from_ref, store_wav, save_upload) -> None:
    _ctx.update(wav_dir=Path(wav_dir), tmp_dir=Path(tmp_dir), model_dir=Path(model_dir), ir_dir=Path(ir_dir),
                usage_file=Path(usage_file), safe_name=safe_name, safe_subfolder=safe_subfolder,
                unique_file=unique_file, wav_rel=wav_rel, wav_from_ref=wav_from_ref, store_wav=store_wav,
                save_upload=save_upload)


def session() -> EngineSession:
    global _session
    if _session is None:
        _session = EngineSession(_ctx["tmp_dir"] / "capture")
    return _session


def shutdown() -> None:
    if _session is not None:
        _session.shutdown()


# ------------------------------------------------------------------------------------ errors

def fail(status: int, key: str, **params) -> HTTPException:
    return HTTPException(status, tr(key, **params))


def engine_error(e: EngineError) -> HTTPException:
    text = tr("cap.err." + e.key, detail=e.detail)
    if text.startswith("cap.err."):
        text = tr("cap.err.internal", detail=f"{e.key}: {e.detail}")
    return HTTPException(409 if e.key == "busy" else 400, text)


# --------------------------------------------------------------------------------- helpers

def _opened() -> dict:
    opened = session().opened
    if not opened:
        raise fail(409, "srv.cap.not_open")
    return opened


def _single_input(opened: dict) -> dict:
    """A capture / level test measures whatever input is FIRST in the open list - so the interface must be open on exactly the one
    RETURN input chosen in step 1. Anything else (e.g. the guitar+return pair a stopped live session leaves behind) would silently
    record the wrong channel, so it is refused instead."""
    if len(opened["in_channels"]) != 1:
        raise fail(409, "srv.cap.bad_open_channels")
    return opened


def _channel_of(opened: dict, kind: str) -> int:
    return opened["out_channels" if kind == "out" else "in_channels"][0]


def _calibrations(opened: Optional[dict]) -> dict:
    if not opened:
        return {"out": None, "in": None, "lin": None}
    name = _device_name(opened)
    out = store.get("out", name, _channel_of(opened, "out"), RATE)
    inp = store.get("in", name, _channel_of(opened, "in"), RATE)
    if inp is not None:
        inp = {**inp, "stale": store.is_stale(inp, None if out is None else out.get("co_dbu"))}
    lin = store.get("lin", name, _channel_of(opened, "in"), RATE)
    if lin is not None:
        # a linearity test says nothing about an input that was calibrated again (or gained since) after it
        lin = {**lin, "stale": inp is None or inp["date"] > lin["date"] or abs(inp["ci_dbu"] - lin["ci_ref"]) > store.STALE_CO_DB}
    return {"out": out, "in": inp, "lin": lin}


_device_names: dict = {}


def _device_name(opened: dict) -> str:
    """The name calibrations are filed under. It comes from the open device itself: the id (an index into the OS device list) can
    point to another interface after something is plugged in or out, so it is only a last resort."""
    return opened.get("device_name") or _device_names.get(opened["device"], opened["device"])


def _int_list(values, what: str) -> List[int]:
    try:
        out = [int(v) for v in values]
    except (TypeError, ValueError):
        raise fail(400, "srv.cap.bad_channels")
    if not out or any(v < 0 for v in out):
        raise fail(400, "srv.cap.bad_channels")
    return out


def describe_input_cached(path: Path) -> dict:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _input_cache:
        _input_cache[key] = al.describe_input(path)
    return _input_cache[key]


def find_v3_inputs() -> List[dict]:
    """The standard v3 input files in the WAV library (identified by their md5, like the trainer does)."""
    found = []
    wav_dir = _ctx["wav_dir"]
    for dirpath, dirnames, filenames in os.walk(wav_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "_"))]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() != ".wav":
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if not 20_000_000 < size < 40_000_000:                # the v3 input is about 27 MB (24 bit, mono, 190 s)
                continue
            try:
                if describe_input_cached(path)["v3"]:
                    found.append({"rel": path.relative_to(wav_dir).as_posix(), "name": name})
            except Exception:
                continue
    found.sort(key=lambda f: (not f["rel"].startswith(INPUT_FOLDER + "/"), f["rel"].lower()))   # the Input folder first
    return found


def _public_engine(snap: dict) -> dict:
    result = snap.get("result")
    if result:
        result = {k: v for k, v in result.items() if k != "trace"}
    return {**snap, "result": result}


# --------------------------------------------------------------------------------- native engine relay

NATIVE_HOST, NATIVE_PORT = "127.0.0.1", 8767   # packaging/ControlServer.swift, started by the app shell alongside us


def _native_call(method: str, path: str, body: Optional[dict] = None, timeout: float = 15.0) -> dict:
    """A small relay to the native engine's own local control server (see packaging/ControlServer.swift). Stdlib
    only (no extra dependency): a handful of short-lived local requests, nothing that needs a real HTTP client."""
    import http.client

    conn = http.client.HTTPConnection(NATIVE_HOST, NATIVE_PORT, timeout=timeout)
    try:
        payload = json.dumps(body or {})
        conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read() or b"{}")
    except (OSError, ValueError) as e:
        raise fail(502, "srv.cap.native_unreachable", detail=f"{type(e).__name__}: {e}")
    finally:
        conn.close()
    if resp.status >= 400:
        raise fail(400, "srv.cap.native_error", detail=data.get("error", str(resp.status)))
    return data


class _State:
    """The few pieces of state more than one module needs (each was a module global while this was one file)."""
    take: Optional[dict] = None           # the take being recorded / analysed / waiting for a decision
    native_prev: Optional[dict] = None    # device/channels Python had open before handing it to the native engine
    live_active: Optional[dict] = None    # {"device_name","out_ch","return_ch","calibrated"} while a live session runs -
                                          # lets native_live_swap recompute gains without the device open in Python
    tuner_active: Optional[dict] = None   # the tuner running on its own on the native engine (capture_api/tuner.py): {"guitar_ch"}
    player_active: Optional[dict] = None  # the same for the NAM player (capture_api/player.py): which device/channels, the
                                          # calibrated numbers its gains come from, which leg is heard


st = _State()
