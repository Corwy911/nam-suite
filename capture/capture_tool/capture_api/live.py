"""The "Prova dal vivo" tool: NAM model + IR, live, on the native engine - file library, uploads, hot-swap, Edit NAM controls."""

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import fsutil
from capture import calibration as cal
from capture import store
from capture.session import EngineError

from .core import (
    RATE,
    _channel_of,
    _ctx,
    _device_name,
    _native_call,
    _opened,
    engine_error,
    fail,
    session,
    st,
)

router = APIRouter()


# --------------------------------------------------------------------- "Prova dal vivo" (NAM + IR, native)
#
# Roman doesn't want files moved into the app's own library just to try them: a file picked with "Sfoglia" is
# used from wherever it is, for that session only, unless "Importa nell'app" is ticked. Three kinds of reference
# a `model`/`ir` field can hold:
#   - a plain filename           -> already in MODEL_DIR/IR_DIR (picked from the dropdown, or a previous import)
#   - "tmp:<token><ext>"         -> a one-off upload (browsed, not imported), living under tmp_dir/live_uploads/,
#                                    kept until the 24 h sweep in app.py (a second Avvia / a later swap needs it again)
# /native/live/upload is the only way a `tmp:` reference is created; the page never invents one on its own.

_LIVE_EXT = {"model": ".nam", "ir": ".wav"}


def _live_dir(kind: str) -> Path:
    return _ctx["model_dir"] if kind == "model" else _ctx["ir_dir"]


def _live_tmp_dir() -> Path:
    d = _ctx["tmp_dir"] / "live_uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_child(base: Path, filename: str) -> Path:
    """Resolves `filename` strictly inside `base` - no subfolders, no `..`."""
    if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
        raise fail(400, "srv.cap.bad_filename")
    path = (base / filename).resolve()
    if path.parent != base.resolve():
        raise fail(400, "srv.cap.bad_filename")
    return path


def _resolve_live_ref(kind: str, ref: str) -> Path:
    if ref.startswith("tmp:"):
        return _safe_child(_live_tmp_dir(), ref[len("tmp:"):])
    return _safe_child(_live_dir(kind), ref)


def _load_usage() -> dict:
    try:
        data = json.loads(_ctx["usage_file"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("model", {})
    data.setdefault("ir", {})
    return data


def _bump_usage(kind: str, name: str) -> None:
    """Only for a saved (library) file, never a one-off upload - see native_live_start."""
    data = _load_usage()
    data[kind][name] = data[kind].get(name, 0) + 1
    fsutil.write_text_atomic(_ctx["usage_file"], json.dumps(data, indent=1))


def _forget_usage(kind: str, name: str) -> None:
    data = _load_usage()
    if data[kind].pop(name, None) is not None:
        fsutil.write_text_atomic(_ctx["usage_file"], json.dumps(data, indent=1))


@router.get("/native/live/files")
def native_live_files():
    """The .nam profiles and cabinet IRs saved in the app's own folders (imported earlier, or dropped in by
    hand) - not the only way to use one, see /native/live/upload. Sorted by how often each has been used for
    a live session, most first ("le salvate che uso di più")."""
    usage = _load_usage()

    def listing(kind: str, ext: str) -> List[dict]:
        names = sorted(p.name for p in _live_dir(kind).glob(f"*{ext}") if p.is_file())
        return sorted(({"name": n, "uses": usage[kind].get(n, 0)} for n in names), key=lambda e: (-e["uses"], e["name"]))

    return {"models": listing("model", ".nam"), "irs": listing("ir", ".wav"),
            "model_dir": str(_ctx["model_dir"]), "ir_dir": str(_ctx["ir_dir"])}


@router.post("/native/live/files/reveal")
def native_live_files_reveal(kind: str):
    """"Apri cartella" next to Sfoglia: opens Modelli NAM/IR directly in the OS file manager, so a whole batch of
    files can be dropped in at once instead of importing them one by one - same idea as captures.py's own
    captures_reveal, same folder /native/live/files already lists from."""
    if kind not in _LIVE_EXT:
        raise fail(400, "srv.cap.bad_channels")
    d = _live_dir(kind)
    d.mkdir(parents=True, exist_ok=True)
    cmd = ["open", str(d)] if sys.platform == "darwin" else ["explorer", str(d)] if sys.platform.startswith("win") else ["xdg-open", str(d)]
    try:
        subprocess.Popen(cmd)
    except OSError as e:
        raise fail(500, "srv.cap.reveal_failed", detail=str(e))
    return {"path": str(d)}


@router.delete("/native/live/files")
def native_live_files_delete(kind: str, name: str):
    """Removes a saved model/IR from the library ("elimina" on right-click) - the file on disk and its usage
    count. Never touches a one-off upload (those aren't in the library, and are already gone by the time this
    could be called - see native_live_start's cleanup)."""
    if kind not in _LIVE_EXT:
        raise fail(400, "srv.cap.bad_channels")
    path = _safe_child(_live_dir(kind), name)
    removed = path.is_file()
    if removed:
        path.unlink()
    _forget_usage(kind, name)
    return {"removed": removed}


@router.post("/native/live/upload")
def native_live_upload(kind: str = Form(...), do_import: bool = Form(False), file: UploadFile = File(...)):
    """A file picked with the page's own "Sfoglia" button (a plain browser file input - it can point at any
    folder on disk, this is not a folder browser on the server). do_import=false (the default): used once, from
    a throwaway copy under tmp/ that native_live_start deletes right after the engine has read it - Roman's own
    file never moves. do_import=true: kept for good in MODEL_DIR/IR_DIR, same as any other saved model/IR."""
    if kind not in _LIVE_EXT:
        raise fail(400, "srv.cap.bad_channels")
    ext = _LIVE_EXT[kind]
    if not (file.filename or "").lower().endswith(ext):
        raise fail(400, "srv.cap.live_bad_extension", ext=ext)
    if do_import:
        stem = _ctx["safe_name"](Path(file.filename).stem, fallback=kind)
        dest = _ctx["unique_file"](_live_dir(kind) / f"{stem}{ext}")
        _ctx["save_upload"](file, dest)
        return {"kind": kind, "ref": dest.name, "imported": True}
    dest = _live_tmp_dir() / f"{uuid.uuid4().hex}{ext}"
    _ctx["save_upload"](file, dest)
    return {"kind": kind, "ref": f"tmp:{dest.name}", "imported": False}


DEFAULT_MASTER_DB = -18.0         # the listening volume of the live tool and of the player: added to the output gain BEFORE the engine's ceiling, so a hot
MASTER_RANGE_DB = (-60.0, 12.0)   # NAM profile does not clip inside the chain; it starts at -18 dB every time (the page's "Volume di ascolto" knob)


def master_clamp(db: float) -> float:
    return max(MASTER_RANGE_DB[0], min(MASTER_RANGE_DB[1], db))


class NativeLiveBody(BaseModel):
    confirm: bool = False
    master_db: float = DEFAULT_MASTER_DB   # the listening volume, on top of the output gain (start only: later changes go through /native/live/controls)
    model: Optional[str] = None   # a ref from /native/live/files or /native/live/upload, or None for IR-only ("testata")
    ir: Optional[str] = None      # same, or None
    listen_out_channels: Optional[List[int]] = None    # single-chain mode (Testata, or Pedale without "insieme"): the 1-2 device
    # channels the result is heard on - the monitor outputs chosen in step 1 - instead of the reamp/amp output (out_ch), which
    # only supplies Co for the gain maths. None keeps the old behaviour (out_ch).
    monitor_out_channels: Optional[List[int]] = None   # 1 or 2 channels -> "insieme" mode: chitarra -> uscita del
    # passo 1 a guadagno unitario (pilota il pedale vero), CONTEMPORANEAMENTE a ritorno -> NAM -> IR -> qui (Roman:
    # "devo poterlo ascoltare mentre sta pilotando il pedale"). None (the default) -> today's single-chain mode,
    # unaffected. Only native_live_start reads this; native_live_swap never needs it (the device stays open).
    calibrated: bool = True       # False: skip requiring the output/return (and, in "insieme" mode, guitar) input
    # calibrations - pre_gain_db/gain_db (and guitar_gain_db) are then always 0 dB, always startable, just less
    # precise. native_live_swap has no field of its own for this: it keeps whatever native_live_start chose, from
    # st.live_active.


def _resolve_live_refs(model_ref: Optional[str], ir_ref: Optional[str]):
    """Resolves and validates both refs before anything touches the native engine or Python's own device -
    a 404 here must never have closed/opened anything (see the tests)."""
    if not model_ref and not ir_ref:
        raise fail(400, "srv.cap.live_need_model_or_ir")
    model_path = _resolve_live_ref("model", model_ref) if model_ref else None
    ir_path = _resolve_live_ref("ir", ir_ref) if ir_ref else None
    for path, ref in ((model_path, model_ref), (ir_path, ir_ref)):
        if path is not None and not path.is_file():
            # a "Sfoglia" pick that was never imported only lives as long as the throwaway copy under tmp/ -
            # if the server restarted since (the packaged app relaunching for the native engine, the test
            # server restarting) that copy is gone: say so plainly instead of showing its meaningless hash name
            if ref.startswith("tmp:"):
                raise fail(404, "srv.cap.live_upload_expired")
            raise fail(404, "srv.cap.live_file_missing", name=path.name)
    return model_path, ir_path


def _apply_live_files(model_path: Optional[Path], ir_path: Optional[Path], model_ref: Optional[str],
                       ir_ref: Optional[str], out_cal: Optional[dict], in_cal: Optional[dict], master_db: float = DEFAULT_MASTER_DB) -> dict:
    """Loads/unloads model_path/ir_path on the ALREADY-OPEN native engine and (re)computes both gains from
    out_cal/in_cal - shared by native_live_start (right after /open) and native_live_swap (while a session is
    already running: hot-swapping the model/IR without stopping playback is exactly what rt_load_model/
    rt_load_ir are for, see RTEngine.cpp's own comment on the point). Deletes a one-off upload's throwaway
    copy once the engine has read it, and counts a saved (library) file's use, either way.
    out_cal/in_cal are the output's and the RETURN input's calibrations, either may be None: for an uncalibrated session
    (NativeLiveBody.calibrated=False) both are None and both gains are 0 dB, regardless of what the model file itself claims."""
    model_in = model_out = None
    if model_path is not None:
        levels = _native_call("POST", "/model", {"path": str(model_path)})
        model_in, model_out = levels.get("input_level_dbu"), levels.get("output_level_dbu")
    else:
        _native_call("POST", "/model/unload")
    if ir_path is not None:
        _native_call("POST", "/ir", {"path": str(ir_path)})
    else:
        _native_call("POST", "/ir/unload")
    # each gain uses only what it needs, so a missing return calibration does not spoil the output gain and the other way round:
    # pre-model gain: the return's Ci minus the profile's own input level; output gain: the profile's output level minus Co, or - with
    # no such level (IR alone) - the return's Ci minus Co when both are known. Anything not computable stays at 0 dB.
    # (Both used to have the opposite sign: a transparent capture then played back 3x too loud - see test_capture_math.)
    pre_gain_db = cal.nam_input_gain_db(model_in, in_cal["ci_dbu"]) if (in_cal is not None and model_in is not None) else 0.0
    if out_cal is not None and model_out is not None:
        gain_db = cal.nam_output_gain_db(model_out, out_cal["co_dbu"])
    elif out_cal is not None and in_cal is not None and model_out is None:
        gain_db = cal.unity_gain_db(in_cal["ci_dbu"], out_cal["co_dbu"])
    else:
        gain_db = 0.0
    _native_call("POST", "/pregain", {"pre_gain_db": pre_gain_db})
    _native_call("POST", "/gain", {"gain_db": gain_db + master_db})       # the gain the page shows stays the calibrated one: the master is added on the way out
    # one-off "Sfoglia" copies are deliberately NOT deleted here (nor on stop): the pickers keep pointing at them, so a
    # second Avvia or a later swap needs them again; the 24 h sweep in app.py's clean_tmp() removes them
    if model_path is not None and not model_ref.startswith("tmp:"):
        _bump_usage("model", model_path.name)
    if ir_path is not None and not ir_ref.startswith("tmp:"):
        _bump_usage("ir", ir_path.name)
    return {"pre_gain_db": round(pre_gain_db, 2), "gain_db": round(gain_db, 2),
            "model_input_level_dbu": model_in, "model_output_level_dbu": model_out}


@router.post("/native/live/start")
def native_live_start(body: NativeLiveBody):
    """NAM (amp profile) + IR (cabinet), live, from the RETURN input (#returnCh on the page) - not the guitar
    input, UNLESS body.monitor_out_channels is set: then both run together (manual chapter 6/10):
      - testata: IR only (body.model=None) - swap cabinets on a real head's mic/DI return without a capture.
      - pedale, solo ritorno: NAM + IR from whatever is already feeding the return (a pedal driven some other
        way, or a head's own return) - same as testata but with a model too.
      - pedale, insieme (monitor_out_channels set): the guitar drives the pedal at unity gain THE WHOLE TIME
        (opened["in_channels"][0]), while the pedal's return (opened["in_channels"][1]) goes through NAM+IR to
        a SEPARATE monitor output - so the loop never breaks while you listen and correct the NAM/IR choice
        (see native_live_swap). Needs Python's own device already open with BOTH input channels, guitar first.
    Gain math: the pre-NAM gain feeds the model at ITS OWN calibrated input level (from the .nam file, if it
    carries one); the final gain always lands the chain on the interface's real calibrated output level - so a
    correctly-calibrated model plays back at the same relative loudness it would in a capture. The guitar leg
    (insieme mode only) reuses the exact same G = Ci - Co as the guitar leg's unity gain. body.calibrated=False
    skips needing any of that calibration data at all (both gains land at 0 dB instead) - always startable, for
    trying a pedal/model before steps 3-4 are even done. To change the model/IR
    once this is already running, see native_live_swap - not this endpoint again; for input/output/gate/EQ
    trims on an already-running session, see native_live_controls."""
    release_tuner(restore=True)                    # these start from Python's open device: it has to be there
    opened = _opened()
    if not body.confirm:
        raise fail(400, "srv.cap.confirm_needed")
    if st.player_active is not None:
        raise fail(409, "srv.cap.player_running")
    if not body.model and not body.ir:
        raise fail(400, "srv.cap.live_need_model_or_ir")
    dual = body.monitor_out_channels is not None
    if body.listen_out_channels is not None and (len(body.listen_out_channels) not in (1, 2) or any(c < 0 for c in body.listen_out_channels)):
        raise fail(400, "srv.cap.live_bad_monitor_channels")
    if dual and len(body.monitor_out_channels) not in (1, 2):
        raise fail(400, "srv.cap.live_bad_monitor_channels")
    name = _device_name(opened)
    out_ch = _channel_of(opened, "out")
    if dual and out_ch in body.monitor_out_channels:      # the pedal send and the monitor on one channel: the monitor would overwrite the send
        raise fail(400, "srv.cap.out_clash")
    out = store.get("out", name, out_ch, RATE) if body.calibrated else None
    guitar_ch = guitar_cal = None
    if dual:
        if len(opened["in_channels"]) < 2:
            raise fail(400, "srv.cap.live_need_both_inputs")
        guitar_ch, return_ch = opened["in_channels"][0], opened["in_channels"][1]
    else:
        return_ch = _channel_of(opened, "in")
    inp = store.get("in", name, return_ch, RATE) if body.calibrated else None
    if body.calibrated:
        # what "calibrated" strictly needs: the guitar leg is exact unity gain (G = Ci_guitar - Co), so it needs the output and the
        # guitar input; the return's own calibration only matters when a NAM profile is loaded (its levels are set against it) -
        # for a head, the return is a mic/DI and IR alone needs no calibration there
        if dual:
            guitar_cal = store.get("in", name, guitar_ch, RATE)
            if out is None or guitar_cal is None:
                raise fail(409, "srv.cap.need_input_cal")
        if body.model and (out is None or inp is None):
            raise fail(409, "srv.cap.need_input_cal")
    model_path, ir_path = _resolve_live_refs(body.model, body.ir)

    blocksize = (opened.get("info") or {}).get("blocksize", 0)
    st.native_prev = {"device": opened["device"],
                     "in_channels": [guitar_ch, return_ch] if dual else [return_ch],
                     "out_channels": [out_ch], "blocksize": blocksize}
    session().close_device()
    try:
        guitar_gain_db = None
        if dual:
            guitar_gain_db = (guitar_cal["ci_dbu"] - out["co_dbu"]) if body.calibrated else 0.0
            _native_call("POST", "/open_dual", {"device": name, "guitar_channel": guitar_ch, "return_channel": return_ch,
                                                  "pedal_out_channel": out_ch, "monitor_out_channels": body.monitor_out_channels,
                                                  "blocksize": blocksize, "guitar_gain_db": guitar_gain_db, "ceiling_db": -1.0})
        else:
            listen = body.listen_out_channels or [out_ch]
            _native_call("POST", "/open", {"device": name, "in_channel": return_ch, "out_channel": listen[0],
                                            "extra_out_channels": listen[1:],
                                            "blocksize": blocksize, "gain_db": 0.0, "ceiling_db": -1.0})
        master = master_clamp(body.master_db)
        result = _apply_live_files(model_path, ir_path, body.model, body.ir, out, inp, master)
    except HTTPException:
        try:
            session().open(**st.native_prev)
        except EngineError:
            pass
        st.native_prev = None
        raise
    st.live_active = {"device_name": name, "out_ch": out_ch, "return_ch": return_ch, "calibrated": body.calibrated, "master_db": master, "post_db": result["gain_db"]}
    resp = {**result, "co_dbu": out["co_dbu"] if out else None, "ci_dbu": inp["ci_dbu"] if inp else None}
    if dual:
        resp["guitar_gain_db"] = round(guitar_gain_db, 2)
    return resp


@router.post("/native/live/swap")
def native_live_swap(body: NativeLiveBody):
    """Changes which NAM model / IR the ALREADY RUNNING live session uses, in place - no stop, no gap in the
    guitar-through-the-pedal-and-back loop. Picked the wrong amp for a pedal mid-session? Change it here
    while still listening, same as native_live_start would but without reopening the device."""
    if st.live_active is None:
        raise fail(409, "srv.cap.live_not_running")
    if not body.confirm:
        raise fail(400, "srv.cap.confirm_needed")
    calibrated = st.live_active.get("calibrated", True)   # keeps whatever native_live_start chose, not body.calibrated
    out = store.get("out", st.live_active["device_name"], st.live_active["out_ch"], RATE) if calibrated else None
    inp = store.get("in", st.live_active["device_name"], st.live_active["return_ch"], RATE) if calibrated else None
    if calibrated and body.model and (out is None or inp is None):      # a profile needs the return's calibration; IR alone does not
        raise fail(409, "srv.cap.need_input_cal")
    model_path, ir_path = _resolve_live_refs(body.model, body.ir)
    result = _apply_live_files(model_path, ir_path, body.model, body.ir, out, inp, st.live_active.get("master_db", DEFAULT_MASTER_DB))
    st.live_active["post_db"] = result["gain_db"]
    return {**result, "co_dbu": out["co_dbu"] if out else None, "ci_dbu": inp["ci_dbu"] if inp else None}


def stop_native(restore: bool = True) -> dict:
    """Stops whatever runs on the native engine (a live session, the player, the tuner) and, with `restore`, hands the device back to Python as it was.
    Every open of the interface is a driver session that sets its one buffer size, and the driver goes back to its own setting in the gaps between
    sessions: callers that open the interface again right after (the page restoring what it had, an open/close request) pass restore=False."""
    result = _native_call("POST", "/stop")
    st.live_active = None
    st.player_active = None
    st.tuner_active = None
    prev, st.native_prev = st.native_prev, None
    if prev is not None and restore:
        try:
            session().open(**prev)
        except EngineError as e:
            raise engine_error(e)
    return result


def release_tuner(restore: bool = False) -> None:
    """The tuner that runs on its own (capture_api/tuner.py) only holds the interface until something else needs it: a live session,
    the player, or the page opening / closing the device itself call this first and get Python's device back as it was."""
    if st.tuner_active is None:
        return
    try:
        stop_native(restore)                      # default: whoever called this opens or closes the device right after, no Python session in between
    except HTTPException:
        st.tuner_active = None                    # the native engine is gone: nothing left to stop
        st.native_prev = None


@router.post("/native/live/stop")
def native_live_stop(restore: bool = True):
    return stop_native(restore)


class NativeLiveControlsBody(BaseModel):
    pre_gain_db: Optional[float] = None     # "Input" knob: on top of native_live_start's own calibrated pre-gain
    gain_db: Optional[float] = None         # "Output" knob: same, for the final output gain
    gate_threshold_db: Optional[float] = None
    eq_bass_db: Optional[float] = None
    eq_mid_db: Optional[float] = None
    eq_treble_db: Optional[float] = None
    ir_in_db: Optional[float] = None        # the IR's own level knobs: gain right before / right after the convolver
    ir_out_db: Optional[float] = None
    master_db: Optional[float] = None       # the listening volume, added to the output gain (live tool and player)


@router.post("/native/live/controls")
def native_live_controls(body: NativeLiveControlsBody):
    """"Edit NAM": the manual controls a real amp-modeler plugin has (input/output level, noise gate, a 3-band
    tone stack - native/engine/RTEngine.cpp) on top of whatever native_live_start/swap already computed from
    calibration. Every field is optional and independent, so the page only ever sends the one control that
    just moved. Requires st.live_active (a running live session) the same way /gain and /pregain always have -
    silently a no-op on the native side if nothing is open, nothing here needs to special-case that itself."""
    if st.live_active is None:
        raise fail(409, "srv.cap.live_not_running")
    a = st.live_active
    if body.master_db is not None:
        a["master_db"] = master_clamp(body.master_db)
    if body.gain_db is not None:
        a["post_db"] = body.gain_db
    master = a.get("master_db", DEFAULT_MASTER_DB)
    # the output gain goes out with the master added (in its usual place among the others); the master alone resends it below
    relay_controls(body.model_copy(update={"gain_db": None if body.gain_db is None else body.gain_db + master}))
    if body.gain_db is None and body.master_db is not None:
        _native_call("POST", "/gain", {"gain_db": a.get("post_db", 0.0) + master})
    return {"ok": True}


def relay_controls(body: NativeLiveControlsBody) -> None:
    """Sends the fields that are set to the native engine - shared by the live tool and the player (same chain, same knobs)."""
    if body.pre_gain_db is not None:
        _native_call("POST", "/pregain", {"pre_gain_db": body.pre_gain_db})
    if body.gain_db is not None:
        _native_call("POST", "/gain", {"gain_db": body.gain_db})
    if body.gate_threshold_db is not None:
        _native_call("POST", "/gate", {"threshold_db": body.gate_threshold_db})
    eq_body = {k: v for k, v in {"bass_db": body.eq_bass_db, "mid_db": body.eq_mid_db, "treble_db": body.eq_treble_db}.items()
               if v is not None}
    if eq_body:
        _native_call("POST", "/eq", eq_body)
    ir_body = {k: v for k, v in {"in_db": body.ir_in_db, "out_db": body.ir_out_db}.items() if v is not None}
    if ir_body:
        _native_call("POST", "/ir_gain", ir_body)
