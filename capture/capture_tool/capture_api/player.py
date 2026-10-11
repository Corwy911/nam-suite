"""The "Lettore NAM" (NAM player): the capture's model played live with the guitar, and an A/B against the real gear.

The loop it closes: capture -> send to the trainer -> back here with the .nam -> load it -> play the guitar through the model (B) and
through the real amp (A, over the same reamp chain the capture used), and switch between them with every level matched by calibration.

Routing (native/engine/RTEngine.h's RTPlayer does the audio, this module only decides the numbers and drives the native engine):

    guitar (Hi-Z) --- feed: Ci_guitar - Co ---------------------------------> reamp output -> reamp box -> amp   (the same unity gain as the monitor)
    guitar (Hi-Z) --- B: Ci_guitar - model input -> NAM -> [IR] -> model output - Co ---+
    return        --- A: [IR] -> Ci_return - Co --------------------------------------+--> A/B switch --> monitor output(s)

"Test Pedali": with a pedal's NAM loaded as well, the pedal is tested in front of the amp's model (the real pedal's output comes back on the return input):

    guitar (Hi-Z) --- feed: Ci_guitar - Co -------------------------------------------------------------> reamp output -> real pedal
    guitar (Hi-Z) --- B: gate -> Ci_guitar - pedal input -> PEDAL NAM -> pedal output - amp input -> AMP NAM -> [IR] -> amp output - Co ---+
    return        --- A: [the real pedal's output] Ci_return - amp input -> AMP NAM (a second copy) -> [IR] -> amp output - Co ---------+--> A/B

The amp stage is identical on both legs, so A/B reveals what the pedal's model gets wrong, and the pedal's output level (output_level_dbu of its
file: the capture of a pedal is calibrated on both sides) is what sets the level that reaches the amp.

Both legs end on the same Co, so the level of the real gear and the level of the model are comparable by construction: what A/B reveals is
what the model gets wrong, not a level difference. Needs the interface opened with both inputs, guitar first (like the live tool's
"together" mode) and, for calibrated levels, the output, the guitar input and the return input calibrated.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from capture import calibration as cal
from capture import store
from capture.session import EngineError
from i18n import tr

from .core import (
    RATE,
    _device_name,
    _native_call,
    _opened,
    fail,
    session,
    st,
)
from .live import (
    DEFAULT_MASTER_DB,
    master_clamp,
    NativeLiveControlsBody,
    _bump_usage,
    _resolve_live_refs,
    release_tuner,
    relay_controls,
    stop_native,
)

router = APIRouter()

MAX_B_DELAY_MS = 190.0      # RTEngine.cpp's kMaxBDelay is 9600 samples = 200 ms at 48 kHz
MANUAL_POST_DB = 0.0        # the model's output when its level is NOT worked out from the calibration (the master volume below is what keeps it safe)
# The listening volume (DEFAULT_MASTER_DB, in live.py: the same one as the live tool's) is added on top of both legs' output gain, BEFORE their ceiling - a NAM
# profile is hot (a guitar at -17 dBFS comes out of a real profile at the -1 dBFS ceiling) and clips inside the leg, where nothing after it could undo it.


class PlayerBody(BaseModel):
    confirm: bool = False
    model: Optional[str] = None          # a ref from /native/live/files, /native/live/upload or /trainer/models/fetch: the amp's NAM (optional when a pedal is tested)
    ir: Optional[str] = None             # the cabinet, on the model's leg; None = no IR
    ir_on_gear: bool = False             # the same IR on the real gear's leg too (a head's DI/load-box return has no cabinet; a mic'd cab already has one)
    monitor_out_channels: Optional[List[int]] = None   # 1 or 2 device channels: where the A/B result is heard (start only)
    calibrated: bool = True              # False: no gain is worked out from the calibration (always startable): Input/Output are set by hand (start only)
    use_model_levels: bool = True        # False: the levels written in the profile are ignored too (manual Input/Output even when calibrated) (start only)
    leg: str = "A"                       # which leg is heard first (start only: later changes go through /native/player/leg)
    master_db: float = DEFAULT_MASTER_DB # the listening volume, added to the output gain of both legs (so A and B stay matched)
    b_delay_ms: float = 0.0              # leg B delayed by this much, to line it up with the real gear's round trip (start only)
    pedal: Optional[str] = None          # "Test Pedali": the pedal's NAM, tested in front of `model` (the amp); None = the plain player


def _model_levels(reply: dict) -> tuple:
    return reply.get("input_level_dbu"), reply.get("output_level_dbu")


def _gains(cals: Optional[dict], model_in, model_out, use_model_levels: bool = True) -> dict:
    """The player's gains in dB from the calibrated numbers (cals = {"co", "ci_guitar", "ci_return"}, or None for an uncalibrated session)
    and the levels written in the profile (ignored with use_model_levels False). What cannot be computed stays at the manual values: 0 dB
    on the way in and on the real gear's leg, MANUAL_POST_DB on the model's output; `model_levels` tells the page whether the model's
    gains come from the profile's levels."""
    if not use_model_levels:
        model_in = model_out = None
    from_levels = model_in is not None and model_out is not None
    if cals is None:
        return {"feed": 0.0, "amp": 0.0, "pre": 0.0, "post": MANUAL_POST_DB, "model_levels": False, "amp_matched": False}
    g = cal.player_gains(cals["ci_guitar"], cals["ci_return"], cals["co"], model_in, model_out)
    out = {k: (0.0 if v is None else round(v, 2)) for k, v in g.items()}
    if g.get("post") is None or g.get("pre") is None:
        out["pre"], out["post"] = 0.0, MANUAL_POST_DB
    out["model_levels"] = from_levels
    out["amp_matched"] = g.get("amp") is not None          # False: the return input is not calibrated, the real gear's leg stays at 0 dB, unmatched
    return out


def _pedal_gains(cals: Optional[dict], pedal_in, pedal_out, amp_in, amp_out, use_model_levels: bool = True, amp_present: bool = True) -> dict:
    """The gains of a pedal test, in dB. The pedal model's output level is what the amp is fed with, so a calibrated test needs it (a pedal is
    captured with the return input calibrated too: that is where `output_level_dbu` comes from):
      pedal_pre   guitar -> pedal NAM input: Ci_guitar - pedal input level
      amp_pre_b   pedal NAM output -> amp NAM input: pedal output level - amp input level
      amp_pre_a   the real pedal's return -> amp NAM input (second copy): Ci_return - amp input level
      post        amp NAM output -> monitor, both legs: amp output level - Co
    Manual (no calibration, or levels ignored): everything 0 dB, MANUAL_POST_DB on the way out.
    Without an amp model (`amp_present` False: the pedal alone, real against model) the amp stage is off: leg B ends on the pedal's output level
    (pedal output - Co) and leg A is a bare return again (Ci_return - Co), each with its own gain."""
    manual = {"feed": 0.0, "pedal_pre": 0.0, "pre": 0.0, "amp_pre_a": 0.0, "post": MANUAL_POST_DB, "amp": MANUAL_POST_DB,
              "model_levels": False, "amp_matched": False, "pedal": True, "shared_amp": amp_present}
    if cals is None or not use_model_levels:
        return manual
    if pedal_in is None or pedal_out is None:
        raise fail(409, "srv.cap.player_pedal_levels")
    out = {**manual, "feed": round(cals["ci_guitar"] - cals["co"], 2), "pedal_pre": round(cals["ci_guitar"] - pedal_in, 2), "model_levels": True}
    if not amp_present:
        out["post"] = round(pedal_out - cals["co"], 2)
        if cals.get("ci_return") is not None:
            out["amp"] = round(cals["ci_return"] - cals["co"], 2)
            out["amp_matched"] = True
        return out
    if amp_in is not None and amp_out is not None:
        out["pre"] = round(pedal_out - amp_in, 2)
        out["post"] = out["amp"] = round(amp_out - cals["co"], 2)
        if cals.get("ci_return") is not None:
            out["amp_pre_a"] = round(cals["ci_return"] - amp_in, 2)
            out["amp_matched"] = True
    else:
        out["model_levels"] = False                          # the amp's own levels are missing: the amp stage stays manual
    return out


def _apply_files(model_path: Optional[Path], ir_path: Optional[Path], model_ref: Optional[str], ir_ref: Optional[str], ir_on_gear: bool,
                 cals: Optional[dict], use_model_levels: bool = True, master_db: float = DEFAULT_MASTER_DB,
                 pedal_path: Optional[Path] = None, pedal_ref: Optional[str] = None, had_pedal: bool = False) -> dict:
    """Loads the model (and the IR, on the legs asked for) into the running player and (re)computes the three calibrated gains from the
    model's own levels - shared by start and swap, like live.py's _apply_live_files. With `pedal_path` the pedal's model goes in front of the
    amp's (see the top of this file); `had_pedal`: one was loaded before, so what it left behind is taken away when there is none now."""
    pedal = pedal_path is not None
    if model_path is not None:
        levels = _native_call("POST", "/model", {"path": str(model_path), "legs": ["B", "A"] if pedal else ["B"]})
    else:                                                    # no amp model: that stage is off (nothing of an earlier choice stays loaded)
        levels = {}
        _native_call("POST", "/model/unload", {"legs": ["B", "A"]})
    model_in, model_out = _model_levels(levels)
    ir_on_gear = ir_on_gear or pedal                         # the amp stage is the same on both legs: the cabinet too
    if ir_path is not None:
        _native_call("POST", "/ir", {"path": str(ir_path), "legs": ["B", "A"] if ir_on_gear else ["B"]})
        if not ir_on_gear:
            _native_call("POST", "/ir/unload", {"legs": ["A"]})     # an IR left on the gear's leg by an earlier choice must go
    else:
        _native_call("POST", "/ir/unload", {"legs": ["A", "B"]})
    if pedal:
        reply = _native_call("POST", "/pedal_model", {"path": str(pedal_path)})
        p_in, p_out = _model_levels(reply)
        g = _pedal_gains(cals, p_in, p_out, model_in, model_out, use_model_levels, amp_present=model_path is not None)
        _native_call("POST", "/pedal_controls", {"pre_db": g["pedal_pre"], "post_db": 0.0})
        _native_call("POST", "/pregain", {"pre_gain_db": g["pre"]})
        _native_call("POST", "/amp_pregain", {"gain_db": g["amp_pre_a"]})
        _native_call("POST", "/gain", {"gain_db": g["post"] + master_db})
        _native_call("POST", "/amp_gain", {"gain_db": g["amp"] + master_db})
        if model_path is not None and not model_ref.startswith("tmp:"):
            _bump_usage("model", model_path.name)
        if not pedal_ref.startswith("tmp:"):
            _bump_usage("model", pedal_path.name)
        if ir_path is not None and not ir_ref.startswith("tmp:"):
            _bump_usage("ir", ir_path.name)
        sizes = int(levels.get("sizes") or 1) if isinstance(levels, dict) else 1
        p_sizes = int(reply.get("sizes") or 1) if isinstance(reply, dict) else 1
        return {"gains": g, "model_input_level_dbu": model_in, "model_output_level_dbu": model_out, "model_sizes": sizes, "model_size": sizes - 1,
                "pedal": {"ref": pedal_ref, "input_level_dbu": p_in, "output_level_dbu": p_out, "sizes": p_sizes, "size": p_sizes - 1}}
    if had_pedal:                                            # back to the plain player: no pedal, and the real gear's leg is a bare return again
        _native_call("POST", "/pedal_model/unload", {})
        if model_path is not None:                           # (without an amp model both legs were just unloaded above)
            _native_call("POST", "/model/unload", {"legs": ["A"]})
        _native_call("POST", "/amp_pregain", {"gain_db": 0.0})
    g = _gains(cals, model_in, model_out, use_model_levels)
    _native_call("POST", "/pregain", {"pre_gain_db": g["pre"]})
    _native_call("POST", "/gain", {"gain_db": g["post"] + master_db})        # the gains the page shows stay the calibrated ones: the master is added on the way out
    _native_call("POST", "/amp_gain", {"gain_db": g["amp"] + master_db})
    if model_path is not None and not model_ref.startswith("tmp:"):
        _bump_usage("model", model_path.name)
    if ir_path is not None and not ir_ref.startswith("tmp:"):
        _bump_usage("ir", ir_path.name)
    sizes = int(levels.get("sizes") or 1) if isinstance(levels, dict) else 1
    # an A2 model (official trainer 0.13: a Lite and a full WaveNet in one file) starts on the full one; the page offers the switch
    return {"gains": g, "model_input_level_dbu": model_in, "model_output_level_dbu": model_out, "model_sizes": sizes, "model_size": sizes - 1, "pedal": None}


def _remember_pedal_offsets(a: dict) -> None:
    """The amp's Input knob moves both legs of a pedal test: leg A keeps its own calibrated offset from leg B's value."""
    g = a["gains"]
    a["amp_pre_offset"] = round(g.get("amp_pre_a", 0.0) - g.get("pre", 0.0), 2) if g.get("shared_amp") else 0.0


def _cals_of(device_name: str, out_ch: int, guitar_ch: int, return_ch: int) -> dict:
    """The calibrations the player needs, or a 409 that names the ones missing: the output and the guitar input are required (they set the send
    to the gear and the whole model leg); the return input is what matches the real gear's leg (A) to them, and without it that leg simply
    stays at 0 dB, unmatched (`ci_return` None, `amp_matched` False in the gains)."""
    out = store.get("out", device_name, out_ch, RATE)
    guitar = store.get("in", device_name, guitar_ch, RATE)
    ret = store.get("in", device_name, return_ch, RATE)
    missing = [name for name, c in (("srv.cap.player_cal_out", out), ("srv.cap.player_cal_guitar", guitar)) if c is None]
    if missing:
        raise fail(409, "srv.cap.player_need_cal", what=", ".join(tr(k) for k in missing))
    return {"co": out["co_dbu"], "ci_guitar": guitar["ci_dbu"], "ci_return": ret["ci_dbu"] if ret is not None else None}


@router.post("/native/player/start")
def native_player_start(body: PlayerBody):
    release_tuner(restore=True)
    opened = _opened()
    if not body.confirm:
        raise fail(400, "srv.cap.confirm_needed")
    if st.live_active is not None or st.player_active is not None:
        raise fail(409, "srv.cap.player_busy")
    if not body.model and not body.pedal:
        raise fail(400, "srv.cap.player_need_model")
    if body.leg not in ("A", "B"):
        raise fail(400, "srv.cap.player_bad_leg")
    monitor = body.monitor_out_channels or []
    if len(monitor) not in (1, 2) or any(c < 0 for c in monitor):
        raise fail(400, "srv.cap.live_bad_monitor_channels")
    if len(opened["in_channels"]) < 2:
        raise fail(400, "srv.cap.player_need_both_inputs")
    name = _device_name(opened)
    out_ch = opened["out_channels"][0]
    if out_ch in monitor:                          # one output channel cannot be both the reamp send and the monitor: the monitor would overwrite it
        raise fail(400, "srv.cap.out_clash")
    guitar_ch, return_ch = opened["in_channels"][0], opened["in_channels"][1]
    cals = _cals_of(name, out_ch, guitar_ch, return_ch) if body.calibrated else None
    model_path = _resolve_live_refs(body.model, None)[0] if body.model else None
    ir_path = _resolve_live_refs(None, body.ir)[1] if body.ir else None
    pedal_path = _resolve_live_refs(body.pedal, None)[0] if body.pedal else None
    delay_samples = int(round(max(0.0, min(MAX_B_DELAY_MS, body.b_delay_ms)) * RATE / 1000.0))

    blocksize = (opened.get("info") or {}).get("blocksize", 0)
    st.native_prev = {"device": opened["device"], "in_channels": [guitar_ch, return_ch], "out_channels": [out_ch], "blocksize": blocksize}
    session().close_device()
    try:
        feed_db = cal.unity_gain_db(cals["ci_guitar"], cals["co"]) if cals else 0.0
        _native_call("POST", "/open_player", {"device": name, "guitar_channel": guitar_ch, "return_channel": return_ch,
                                               "reamp_out_channel": out_ch, "monitor_out_channels": monitor, "blocksize": blocksize,
                                               "feed_gain_db": feed_db, "ceiling_db": -1.0})
        master = master_clamp(body.master_db)
        result = _apply_files(model_path, ir_path, body.model, body.ir, body.ir_on_gear, cals, body.use_model_levels, master, pedal_path, body.pedal)
        _native_call("POST", "/b_delay", {"samples": delay_samples})
        _native_call("POST", "/leg", {"leg": body.leg})
    except Exception:
        # whatever failed (a bad model file, the native engine not there): give the device back to Python as it was
        try:
            session().open(**st.native_prev)
        except EngineError:
            pass
        st.native_prev = None
        raise
    st.player_active = {"device_name": name, "out_ch": out_ch, "guitar_ch": guitar_ch, "return_ch": return_ch,
                        "calibrated": body.calibrated, "use_model_levels": body.use_model_levels, "master_db": master,
                        "post_db": result["gains"]["post"], "amp_db": result["gains"]["amp"], "cals": cals, "leg": body.leg, "ir_on_gear": body.ir_on_gear,
                        "model_ref": body.model, "ir_ref": body.ir, "pedal_ref": body.pedal, "feed_gain_db": round(feed_db, 2), **result}
    _remember_pedal_offsets(st.player_active)
    return {**result, "feed_gain_db": round(feed_db, 2), "leg": body.leg, "b_delay_samples": delay_samples,
            "cal": None if cals is None else {"co_dbu": cals["co"], "ci_guitar_dbu": cals["ci_guitar"], "ci_return_dbu": cals["ci_return"]}}


@router.post("/native/player/swap")
def native_player_swap(body: PlayerBody):
    """Another model and/or IR in the running player, without stopping: the guitar keeps going to the real gear, the A/B keeps its leg."""
    if st.player_active is None:
        raise fail(409, "srv.cap.player_not_running")
    if not body.confirm:
        raise fail(400, "srv.cap.confirm_needed")
    if not body.model and not body.pedal:
        raise fail(400, "srv.cap.player_need_model")
    model_path = _resolve_live_refs(body.model, None)[0] if body.model else None
    ir_path = _resolve_live_refs(None, body.ir)[1] if body.ir else None
    pedal_path = _resolve_live_refs(body.pedal, None)[0] if body.pedal else None
    result = _apply_files(model_path, ir_path, body.model, body.ir, body.ir_on_gear, st.player_active["cals"], st.player_active.get("use_model_levels", True),
                          st.player_active.get("master_db", DEFAULT_MASTER_DB), pedal_path, body.pedal, had_pedal=bool(st.player_active.get("pedal")))
    st.player_active.update(ir_on_gear=body.ir_on_gear, model_ref=body.model, ir_ref=body.ir, pedal_ref=body.pedal, post_db=result["gains"]["post"], amp_db=result["gains"]["amp"], **result)
    _remember_pedal_offsets(st.player_active)
    return result


@router.get("/native/player/active")
def native_player_active():
    """What the player is doing right now, or None: a page opened (or reloaded) while it runs picks it up from here and can stop it."""
    a = st.player_active
    if a is None:
        return {"active": None}
    keep = ("leg", "ir_on_gear", "model_ref", "ir_ref", "feed_gain_db", "gains", "model_input_level_dbu", "model_output_level_dbu", "calibrated", "use_model_levels", "master_db",
            "model_sizes", "model_size", "pedal_ref", "pedal")
    return {"active": {k: a.get(k) for k in keep}}


class LegBody(BaseModel):
    leg: str


@router.post("/native/player/leg")
def native_player_leg(body: LegBody):
    """The A/B switch: "A" = the real gear (through the return), "B" = the model."""
    if st.player_active is None:
        raise fail(409, "srv.cap.player_not_running")
    if body.leg not in ("A", "B"):
        raise fail(400, "srv.cap.player_bad_leg")
    _native_call("POST", "/leg", {"leg": body.leg})
    st.player_active["leg"] = body.leg
    return {"leg": body.leg}


class PlayerControlsBody(NativeLiveControlsBody):
    b_trim_db: Optional[float] = None    # B against A, by ear (calibration already matches them)
    b_delay_ms: Optional[float] = None
    amp_gain_db: Optional[float] = None  # the real gear's leg, absolute (starts at the calibrated Ci_return - Co)
    model_size: Optional[int] = None     # an A2 model: 0 = Lite ... model_sizes-1 = full
    pedal_pre_gain_db: Optional[float] = None   # "Test Pedali": the pedal model's input level / output level / A2 size
    pedal_gain_db: Optional[float] = None
    pedal_size: Optional[int] = None


@router.post("/native/player/controls")
def native_player_controls(body: PlayerControlsBody):
    """The model's knobs (input/output level, gate, tone stack, IR levels: same as "Edit NAM" of the live tool) plus the player's own: the
    B trim, the B delay, the gear leg's gain. Every field optional, the page sends only the one that moved."""
    if st.player_active is None:
        raise fail(409, "srv.cap.player_not_running")
    a = st.player_active
    if body.master_db is not None:
        a["master_db"] = master_clamp(body.master_db)
    if body.gain_db is not None:
        a["post_db"] = body.gain_db
    if body.amp_gain_db is not None:
        a["amp_db"] = body.amp_gain_db
    master = a.get("master_db", DEFAULT_MASTER_DB)
    relay_controls(body.model_copy(update={"gain_db": None}))      # the output gain goes out below, with the master added
    shared_amp = bool(a.get("gains", {}).get("shared_amp"))
    if shared_amp:                                                 # a pedal test with an amp: the amp stage is the same on both legs
        if body.pre_gain_db is not None:
            _native_call("POST", "/amp_pregain", {"gain_db": body.pre_gain_db + a.get("amp_pre_offset", 0.0)})
        if body.gain_db is not None:
            a["amp_db"] = body.gain_db
    if a.get("pedal"):
        pedal_body = {k: v for k, v in (("pre_db", body.pedal_pre_gain_db), ("post_db", body.pedal_gain_db), ("size", body.pedal_size)) if v is not None}
        if pedal_body:
            reply = _native_call("POST", "/pedal_controls", pedal_body)
            if body.pedal_size is not None:
                a["pedal"]["size"] = reply.get("size", body.pedal_size) if isinstance(reply, dict) else body.pedal_size
    if body.model_size is not None:
        reply = _native_call("POST", "/model_size", {"index": body.model_size})
        a["model_size"] = reply.get("index", body.model_size) if isinstance(reply, dict) else body.model_size
    if body.b_trim_db is not None:
        _native_call("POST", "/b_trim", {"trim_db": body.b_trim_db})
    if body.b_delay_ms is not None:
        samples = int(round(max(0.0, min(MAX_B_DELAY_MS, body.b_delay_ms)) * RATE / 1000.0))
        _native_call("POST", "/b_delay", {"samples": samples})
    if body.gain_db is not None or body.master_db is not None:
        _native_call("POST", "/gain", {"gain_db": a.get("post_db", 0.0) + master})
    if body.amp_gain_db is not None or body.master_db is not None or (shared_amp and body.gain_db is not None):
        _native_call("POST", "/amp_gain", {"gain_db": a.get("amp_db", 0.0) + master})
    return {"ok": True}


@router.post("/native/player/stop")
def native_player_stop(restore: bool = True):
    return stop_native(restore)
