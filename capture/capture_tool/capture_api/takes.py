"""The input file (library folder, upload), and the takes: record, analyse, one folder per capture, accept, discard."""

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from capture import alignment as al
from capture import store
from capture.session import EngineError

from .core import (
    CAPTURE_FOLDER,
    EXTRA_TAIL,
    INPUT_FOLDER,
    RATE,
    _channel_of,
    _single_input,
    _ctx,
    _device_name,
    _lock,
    _opened,
    describe_input_cached,
    engine_error,
    fail,
    find_v3_inputs,
    session,
    st,
)

router = APIRouter()


# --------------------------------------------------------------------------------- the input file

def organize_library() -> dict:
    """One-off tidy-up of a library made before inputs and results had their own places (idempotent, cheap when there is
    nothing to do): a v3 input sitting loose in Catture/ goes to Input/, and each loose result WAV in Catture/ (with its
    .json sidecar) gets a folder of its own named after it. Only files directly in Catture/ are touched - never
    anything already in a sub-folder. Returns what it moved, for the log."""
    wav_dir = _ctx["wav_dir"]
    base = wav_dir / CAPTURE_FOLDER
    moved = {"inputs": [], "results": []}
    if not base.is_dir():
        return moved
    for path in sorted(base.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        try:
            is_input = 20_000_000 < path.stat().st_size < 40_000_000 and describe_input_cached(path)["v3"]
        except Exception:
            is_input = False
        if is_input:
            dest_dir = wav_dir / INPUT_FOLDER
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = _ctx["unique_file"](dest_dir / path.name)
            path.rename(dest)
            moved["inputs"].append(dest.name)
            continue
        folder = base / path.stem
        n = 2
        while folder.exists():
            folder = base / f"{path.stem} ({n})"
            n += 1
        folder.mkdir()
        path.rename(folder / path.name)
        sidecar = path.with_suffix(".json")
        if sidecar.is_file():
            sidecar.rename(folder / sidecar.name)
        moved["results"].append(folder.name)
    return moved


@router.get("/input")
def input_status():
    return {"found": find_v3_inputs(), "guide": "https://neural-amp-modeler.readthedocs.io/en/latest/tutorials/gui.html"}


@router.post("/input/upload")
def input_upload(file: UploadFile = File(...)):
    created: List[Path] = []
    rel = _ctx["store_wav"](file, INPUT_FOLDER, created)
    path = _ctx["wav_dir"] / rel
    if not al.describe_input(path)["v3"]:
        for p in created:
            p.unlink(missing_ok=True)
        raise fail(400, "srv.cap.input_not_v3")
    return {"rel": rel}


# ---------------------------------------------------------------------------------------- takes

class RecordBody(BaseModel):
    gear: str = "head"                 # head | pedal
    name: str = ""
    input_ref: str
    calibrated: bool = True
    send_gain_db: float = 0.0          # trim of the played file; 0 = the level that was calibrated
    warmup: bool = False               # blips before the input until the driver's one-sample step is over (alignment.WarmupWatch)


RAW_SUBFOLDER = "_originale"        # underscore: skipped by the library scan (find_v3_inputs), so the raw take never shows up as a WAV to pick


def _new_take_dir(take: dict) -> Path:
    """One folder per capture, created the moment the recording ends (not when it is accepted): everything needed to
    recover or re-send it - the raw recording, the aligned WAV, the levels, the calibration it was made with - lives there
    even if the take is discarded or the page is closed before saving."""
    label = _ctx["safe_name"](take["name"] or "Capture", fallback="Capture")
    stamp = time.strftime("%Y-%m-%d %H.%M", time.localtime(take["created"]))
    base = _ctx["wav_dir"] / CAPTURE_FOLDER
    d = base / f"{stamp} - {label}"
    n = 2
    while d.exists():
        d = base / f"{stamp} - {label} ({n})"
        n += 1
    d.mkdir(parents=True)
    return d


def _write_capture_json(take: dict, **updates) -> None:
    """capture.json: the whole record of this capture. Merged into take["_capture"] so accept/discard only add to it."""
    d = take.get("_dir")
    if not d:
        return
    rec = take.setdefault("_capture", {})
    rec.update(updates)
    try:
        (Path(d) / "capture.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _save_take_folder(take: dict, raw: np.ndarray, report: dict) -> None:
    d = _new_take_dir(take)
    take["_dir"] = str(d)
    (d / RAW_SUBFOLDER).mkdir()
    sf.write(str(d / RAW_SUBFOLDER / "raw-recording.wav"), np.asarray(raw, dtype=np.float32), RATE, subtype="FLOAT")
    ctx = take.get("_context", {})
    _write_capture_json(
        take, format=1, status="pending", name=take["name"] or "Capture",
        created=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(take["created"])),
        gear=take["gear"], calibrated=take["calibrated"], send_gain_db=take["send_gain_db"],
        metadata=take["metadata"], warnings=take["warnings"], report=report,
        input=ctx.get("input"), device=ctx.get("device"), calibration=ctx.get("calibration"),
        files={"raw": f"{RAW_SUBFOLDER}/raw-recording.wav", "output": None})


def _new_take(**fields) -> dict:
    return {"id": uuid.uuid4().hex[:10], "status": "recording", "created": time.time(), "report": None, "error": None, **fields}


@router.post("/record/start")
def record_start(body: RecordBody):
    opened = _single_input(_opened())
    s = session()
    if body.gear not in ("head", "pedal"):
        raise fail(400, "srv.cap.bad_gear")
    if not -30.0 <= body.send_gain_db <= 0.0:
        raise fail(400, "srv.cap.level_range")
    path = _ctx["wav_from_ref"](body.input_ref)
    if not describe_input_cached(path)["v3"]:
        raise fail(400, "srv.cap.input_not_v3")
    name = _device_name(opened)
    warnings = []
    metadata = {"input_level_dbu": None, "output_level_dbu": None, "gear_type": "amp" if body.gear == "head" else "pedal"}
    if body.calibrated:
        out = store.get("out", name, _channel_of(opened, "out"), RATE)
        if out is None:
            raise fail(409, "srv.cap.need_output_cal")
        metadata["input_level_dbu"] = round(out["co_dbu"] + body.send_gain_db, 2)     # 0 dBFS of the played file = Co + the trim
        if body.gear == "pedal":
            inp = store.get("in", name, _channel_of(opened, "in"), RATE)
            if inp is None:
                raise fail(409, "srv.cap.need_input_cal")
            metadata["output_level_dbu"] = inp["ci_dbu"]
            if store.is_stale(inp, out["co_dbu"]):
                warnings.append("stale_input_cal")
    with _lock:
        if st.take is not None and st.take["status"] in ("recording", "analyzing"):
            raise fail(409, "srv.cap.take_busy")
        take = _new_take(gear=body.gear, name=body.name.strip(), input_ref=body.input_ref, calibrated=body.calibrated,
                         metadata=metadata, warnings=warnings, send_gain_db=body.send_gain_db)
        take["_raw"] = str(_ctx["tmp_dir"] / f"take-{take['id']}.npy")
        idesc = describe_input_cached(path)
        take["_context"] = {
            "input": {"ref": body.input_ref, "md5": idesc.get("md5"), "frames": idesc.get("frames")},
            "device": {"name": name, "in_channel": _channel_of(opened, "in"), "out_channel": _channel_of(opened, "out"),
                       "rate": RATE, **{k: v for k, v in (opened.get("info") or {}).items() if k in ("blocksize", "latency_in", "latency_out")}},
            "calibration": {"out": store.get("out", name, _channel_of(opened, "out"), RATE),
                            "in": store.get("in", name, _channel_of(opened, "in"), RATE)}}
        st.take = take
    try:
        s.start_op("record", input_path=str(path), save_path=take["_raw"], out_index=0, in_index=0, gain_db=body.send_gain_db,
                   extra_frames=EXTRA_TAIL, warmup=body.warmup)
    except EngineError as e:
        with _lock:
            st.take = None
        raise engine_error(e)
    threading.Thread(target=_finish_take, args=(take,), daemon=True, name="take-analysis").start()
    return {"id": take["id"], "metadata": metadata, "warnings": warnings}


def _finish_take(take: dict) -> None:
    """Waits for the recording, then aligns / cuts / checks it (numpy, a few seconds for 190 s of audio)."""
    s = session()
    try:
        done = s.wait_op(60 * 20)
        result = done["result"]
        if result.get("aborted") or done.get("stopped"):
            take["status"] = "aborted"
            return
        take["status"] = "analyzing"
        rec = np.load(take["_raw"])
        aligned, report = al.analyze_take(rec, al.INPUT_V3_FRAMES, timing=result.get("timing"))
        report["xruns"] = result.get("xruns", 0)
        if report["xruns"]:
            report["warnings"].append("xruns")
            report["ok"] = False
        aligned_path = Path(take["_raw"]).with_name(f"take-{take['id']}-aligned.npy")
        np.save(aligned_path, aligned.astype(np.float32))
        take["_aligned"] = str(aligned_path)
        try:
            _save_take_folder(take, rec, report)
        except Exception:                                 # a full disk must not lose the take that is still in memory
            take["_dir"] = None
        take["report"] = report
        take["status"] = "ready"
    except EngineError as e:
        take["status"], take["error"] = "error", {"key": e.key, "detail": e.detail}
    except Exception as e:                                # analysis failure: never leave the page waiting
        take["status"], take["error"] = "error", {"key": "internal", "detail": f"{type(e).__name__}: {e}"}
    finally:
        try:
            Path(take["_raw"]).unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/record/stop")
def record_stop():
    session().stop_op()
    return {}


class AcceptBody(BaseModel):
    name: str = ""
    folder: str = ""
    force: bool = False               # accept a take the checks flagged (the user looked at the report)


@router.post("/take/accept")
def take_accept(body: AcceptBody):
    """Writes the aligned take as a mono 24-bit 48 kHz WAV into the library and returns what the trainer form needs."""
    with _lock:
        take = st.take
        if take is None or take["status"] != "ready":
            raise fail(409, "srv.cap.no_take")
        if not take["report"]["ok"] and not body.force:
            raise fail(409, "srv.cap.take_flagged")
    label = _ctx["safe_name"](body.name.strip() or take["name"] or "Capture", fallback="Capture")
    folder = _ctx["safe_subfolder"](body.folder.strip() or CAPTURE_FOLDER)
    if take.get("_dir") and not body.folder.strip():
        target_dir = Path(take["_dir"])                     # the capture's own folder (see _new_take_dir)
    else:
        target_dir = _ctx["wav_dir"] / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = _ctx["unique_file"](target_dir / f"{label}.wav")
    samples = np.load(take["_aligned"]).astype(np.float64)
    sf.write(str(dest), samples, RATE, subtype="PCM_24")
    info = sf.info(str(dest))
    if info.frames != al.INPUT_V3_FRAMES or info.channels != 1 or info.samplerate != RATE:
        dest.unlink(missing_ok=True)
        raise fail(500, "srv.cap.write_failed")
    output_rel = _ctx["wav_rel"](dest)
    handoff = {"input_rel": take["input_ref"], "output_rel": output_rel, "name": dest.stem, "metadata": take["metadata"],
               "report": take["report"]}
    # The levels must not depend on the page that made the take: they travel with the WAV (to the trainer, over the LAN or on a disk).
    dest.with_suffix(".json").write_text(json.dumps({k: handoff[k] for k in ("name", "input_rel", "metadata", "report")}, indent=1,
                                                    ensure_ascii=False), encoding="utf-8")
    Path(take["_aligned"]).unlink(missing_ok=True)
    _write_capture_json(take, status="accepted", name=dest.stem, forced=bool(body.force and not take["report"]["ok"]),
                        files={**take.get("_capture", {}).get("files", {}), "output": output_rel})
    handoff["folder_rel"] = _ctx["wav_rel"](Path(take["_dir"])) if take.get("_dir") else None
    with _lock:
        st.take = None
    return handoff


@router.post("/captures/reveal")
def captures_reveal(folder: str = "captures"):
    """"Sfoglia catture" / "Apri cartella input": opens that folder in the OS file manager (this tool only ever serves this PC)."""
    d = _ctx["wav_dir"] / (INPUT_FOLDER if folder == "input" else CAPTURE_FOLDER)
    d.mkdir(parents=True, exist_ok=True)
    cmd = ["open", str(d)] if sys.platform == "darwin" else ["explorer", str(d)] if sys.platform.startswith("win") else ["xdg-open", str(d)]
    try:
        subprocess.Popen(cmd)
    except OSError as e:
        raise fail(500, "srv.cap.reveal_failed", detail=str(e))
    return {"path": str(d)}


@router.post("/take/discard")
def take_discard():
    with _lock:
        take, st.take = st.take, None
    if take is not None:
        _write_capture_json(take, status="discarded")      # the raw recording stays in its folder, on purpose
        for key in ("_raw", "_aligned"):
            if take.get(key):
                Path(take[key]).unlink(missing_ok=True)
    return {}
