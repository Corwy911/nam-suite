"""Interface, calibration (output, input/return, linearity), level test, latency, meter: the wizard's steps 1-4."""

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from capture import calibration as cal
from capture import store
from capture.session import EngineError

from .live import release_tuner
from .core import (
    RATE,
    _calibrations,
    _channel_of,
    _single_input,
    _device_name,
    _device_names,
    _int_list,
    _lock,
    _opened,
    _public_engine,
    engine_error,
    fail,
    session,
    st,
)

router = APIRouter()


# ----------------------------------------------------------------------------------- devices

@router.get("/devices")
def devices():
    s = session()
    try:
        s.ensure_started()
        found = s.call("devices", timeout=90)
    except EngineError as e:
        raise engine_error(e)
    for d in found["devices"]:
        _device_names[d["id"]] = d["name"]
    return found


class OpenBody(BaseModel):
    device: str
    in_channels: List[int]
    out_channels: List[int]
    blocksize: int = 0
    probe: bool = True           # False: no test stream (the page is about to hand the interface to the native engine: one driver session less)


@router.post("/open")
def open_device(body: OpenBody):
    release_tuner()
    s = session()
    ins, outs = _int_list(body.in_channels, "in"), _int_list(body.out_channels, "out")
    try:
        s.ensure_started()
        if s.opened and s.op:
            s.stop_op()
        opened = s.open(body.device, ins, outs, RATE, body.blocksize, body.probe)
    except EngineError as e:
        raise engine_error(e)
    return {**opened, "device_name": _device_name(opened), "calibrations": _calibrations(opened)}


@router.post("/close")
def close_device():
    release_tuner()
    s = session()
    s.stop_op()
    s.close_device()
    return {}


@router.get("/state")
def state():
    s = session()
    snap = s.snapshot()
    opened = snap["opened"]
    with _lock:
        take = None if st.take is None else {k: v for k, v in st.take.items() if not k.startswith("_")}
    return {"engine": _public_engine(snap), "calibrations": _calibrations(opened), "take": take,
            "demo": bool(opened and str(opened["device"]).startswith("sim:"))}


@router.post("/stop")
def stop():
    session().stop_op()
    return {}


# --------------------------------------------------------------------- output level (multimeter)

class ToneBody(BaseModel):
    level_dbfs: float = 0.0
    confirm: bool = False


@router.post("/tone/start")
def tone_start(body: ToneBody):
    _opened()
    if not body.confirm:
        raise fail(400, "srv.cap.confirm_needed")
    if not -60.0 <= body.level_dbfs <= 0.0:
        raise fail(400, "srv.cap.level_range")
    try:
        session().start_op("tone", level_dbfs=body.level_dbfs, out_index=0, in_index=0)
    except EngineError as e:
        raise engine_error(e)
    return {}


class LevelBody(BaseModel):
    level_dbfs: float


@router.post("/tone/level")
def tone_level(body: LevelBody):
    """Changes the level of the tone that is playing (the software trim knob): it glides, it never clicks."""
    _opened()
    if not -60.0 <= body.level_dbfs <= 0.0:
        raise fail(400, "srv.cap.level_range")
    try:
        return session().call("set_level", level_dbfs=body.level_dbfs)
    except EngineError as e:
        raise engine_error(e)


class OutputCalBody(BaseModel):
    vrms: str
    sent_dbfs: float = 0.0
    note: str = ""


@router.post("/calibrate/output")
def calibrate_output(body: OutputCalBody):
    opened = _opened()
    vrms = cal.parse_vrms(body.vrms)
    if vrms is None:
        raise fail(400, "srv.cap.bad_vrms")
    if not -60.0 <= body.sent_dbfs <= 0.0:
        raise fail(400, "srv.cap.level_range")
    co = cal.output_level_dbu(vrms, body.sent_dbfs)
    entry = store.put("out", _device_name(opened), _channel_of(opened, "out"), RATE, co_dbu=round(co, 2), vrms=vrms,
                      sent_dbfs=body.sent_dbfs, note=body.note.strip()[:200])
    return {"entry": entry, "calibrations": _calibrations(opened)}


@router.get("/demo/multimeter")
def demo_multimeter(level_dbfs: float = 0.0):
    """Only with the simulated interface: what a multimeter would read (so the wizard can be tried without hardware)."""
    opened = _opened()
    if not str(opened["device"]).startswith("sim:"):
        raise fail(400, "srv.cap.demo_only")
    return session().call("sim_multimeter", level_dbfs=level_dbfs)


class DemoBody(BaseModel):
    route: Optional[str] = None
    gear: Optional[str] = None
    co_dbu: Optional[float] = None
    ci_dbu: Optional[float] = None
    speed: Optional[float] = None
    latency: Optional[int] = None


@router.post("/demo/config")
def demo_config(body: DemoBody):
    opened = _opened()
    if not str(opened["device"]).startswith("sim:"):
        raise fail(400, "srv.cap.demo_only")
    return session().call("sim", **{k: v for k, v in body.model_dump().items() if v is not None})


# --------------------------------------------------------------- input / return level (loopback)

class InputCalBody(BaseModel):
    note: str = ""
    confirm: bool = False
    level_dbfs: Optional[float] = None      # None: the engine looks for a suitable level; a number: use exactly this software level


@router.post("/calibrate/input")
def calibrate_input(body: InputCalBody):
    """Blocks for the ~10-20 s the search takes; the page keeps polling /state for the meters meanwhile."""
    opened = _opened()
    if not body.confirm:
        raise fail(400, "srv.cap.confirm_needed")
    manual = body.level_dbfs is not None
    if manual and not -60.0 <= body.level_dbfs <= 0.0:
        raise fail(400, "srv.cap.level_range")
    name = _device_name(opened)
    out = store.get("out", name, _channel_of(opened, "out"), RATE)
    if out is None:
        raise fail(409, "srv.cap.need_output_cal")
    s = session()
    try:
        s.start_op("return_cal", co=out["co_dbu"], out_index=0, in_index=0, **({"level_dbfs": body.level_dbfs, "fixed": True} if manual else {}))
        done = s.wait_op(120)
    except EngineError as e:
        raise engine_error(e)
    result = done["result"]
    if result.get("aborted") or done.get("stopped"):
        return {"aborted": True}
    if result["judge"] in ("clip", "silent"):
        # A clipped or absent return says nothing about the input's level: keep the previous calibration, tell the page why.
        return {"entry": None, "rejected": result["judge"], "result": result, "calibrations": _calibrations(opened)}
    entry = store.put("in", name, _channel_of(opened, "in"), RATE, ci_dbu=round(result["ci"], 2), level_dbfs=result["level_dbfs"],
                      peak_dbfs=round(result["peak_dbfs"], 2), spread_db=round(result["spread_db"], 3), judge=result["judge"],
                      manual=manual, co_at_time=out["co_dbu"], note=body.note.strip()[:200])
    return {"entry": entry, "result": result, "calibrations": _calibrations(opened)}


@router.post("/verify/input")
def verify_input():
    """Measures the loopback again at the level used for the calibration and reports how far the input has moved (gain touched?)."""
    opened = _opened()
    name = _device_name(opened)
    out = store.get("out", name, _channel_of(opened, "out"), RATE)
    inp = store.get("in", name, _channel_of(opened, "in"), RATE)
    if out is None or inp is None:
        raise fail(409, "srv.cap.need_input_cal")
    s = session()
    try:
        s.start_op("measure", level_dbfs=inp["level_dbfs"], out_index=0, in_index=0)
        m = s.wait_op(60)["result"]
    except EngineError as e:
        raise engine_error(e)
    ci_now = cal.input_level_dbu(out["co_dbu"], inp["level_dbfs"], m["peak_dbfs"])
    return {"ci_now": round(ci_now, 2), "ci_stored": inp["ci_dbu"], "deviation_db": round(ci_now - inp["ci_dbu"], 2),
            "peak_dbfs": m["peak_dbfs"], "clipped": m["clipped"]}


class LinearityBody(BaseModel):
    confirm: bool = False


@router.post("/linearity")
def linearity(body: LinearityBody):
    """Optional check of the calibrated input at several levels (loopback cabled as for the input calibration). Blocks ~10 s."""
    opened = _opened()
    if not body.confirm:
        raise fail(400, "srv.cap.confirm_needed")
    name = _device_name(opened)
    out = store.get("out", name, _channel_of(opened, "out"), RATE)
    inp = store.get("in", name, _channel_of(opened, "in"), RATE)
    if out is None or inp is None:
        raise fail(409, "srv.cap.need_input_cal")
    s = session()
    try:
        s.start_op("linearity", co=out["co_dbu"], ci=inp["ci_dbu"], out_index=0, in_index=0)
        done = s.wait_op(120)
    except EngineError as e:
        raise engine_error(e)
    result = done["result"]
    if result.get("aborted") or done.get("stopped"):
        return {"aborted": True}
    points = result["points"]
    verdict = cal.linearity_verdict(points)
    rows = [{"level_dbfs": p["level_dbfs"], "read_dbfs": round(p["peak_dbfs"], 2), "ci_dbu": round(p["ci_dbu"], 2), "dev_db": d,
             "thdn_db": round(p["thdn_db"], 1), "clipped": p["clipped"]} for p, d in zip(points, verdict["deviations"])]
    entry = store.put("lin", name, _channel_of(opened, "in"), RATE, status=verdict["status"], reason=verdict["reason"],
                      max_dev_db=verdict["max_dev_db"], median_ci=verdict["median_ci"], span_db=verdict["span_db"],
                      ci_ref=inp["ci_dbu"], co_at_time=out["co_dbu"], points=rows)
    return {"entry": entry, "calibrations": _calibrations(opened)}




class LevelTestBody(BaseModel):
    send_gain_db: float = 0.0
    confirm: bool = False


@router.post("/level-test")
def level_test(body: LevelTestBody):
    """Before a capture, with the gear in the loop: a 1 kHz sine at the level the capture will use (0 dBFS + the trim), and where the
    return peaks. Nothing is stored."""
    _single_input(_opened())
    if not body.confirm:
        raise fail(400, "srv.cap.confirm_needed")
    if not -30.0 <= body.send_gain_db <= 0.0:
        raise fail(400, "srv.cap.level_range")
    s = session()
    try:
        s.start_op("measure", level_dbfs=body.send_gain_db, out_index=0, in_index=0, seconds=3.0)
        m = s.wait_op(60)["result"]
    except EngineError as e:
        raise engine_error(e)
    judge = cal.judge_level(m["sample_peak_dbfs"], m["clipped"])
    return {"level_dbfs": body.send_gain_db, "peak_dbfs": m["sample_peak_dbfs"], "fundamental_dbfs": m["peak_dbfs"],
            "headroom_db": round(-m["sample_peak_dbfs"], 2), "clipped": m["clipped"], "thdn_db": m["thdn_db"], "judge": judge,
            "suggest_trim_db": cal.suggest_trim_db(m["sample_peak_dbfs"]) if judge in ("clip", "hot") else None}


@router.post("/latency-probe")
def latency_probe():
    """Round trip of the interface alone (output cabled to input)."""
    opened = _opened()
    s = session()
    amp = 0.5
    name = _device_name(opened)
    out, inp = store.get("out", name, _channel_of(opened, "out"), RATE), store.get("in", name, _channel_of(opened, "in"), RATE)
    if out and inp:
        # with both levels known, send a click that reads about -6 dBFS at the input (0.5 would overload a sensitive input)
        amp = max(0.01, min(0.5, cal.from_db(-6.0 - (out["co_dbu"] - inp["ci_dbu"]))))
    try:
        s.start_op("click_probe", out_index=0, in_index=0, amp=amp)
        result = s.wait_op(60)["result"]
    except EngineError as e:
        raise engine_error(e)
    return result


@router.post("/meter/start")
def meter_start():
    _opened()
    try:
        session().start_op("meter")
    except EngineError as e:
        raise engine_error(e)
    return {}


@router.get("/calibration")
def calibration_list():
    return {"entries": list(store.all_entries().values())}


@router.delete("/calibration")
def calibration_delete(kind: str, device: str, channel: int):
    return {"removed": store.delete(kind, device, channel, RATE)}
