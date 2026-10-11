"""The tuner on its own: the guitar input (Hi-Z, step 1) listened to by the native engine with the output muted, so notes can be tuned
without starting anything else.

While a live session (with the guitar) or the NAM player runs, the tuner simply reads the note from THEIR engine: the pitch detector
(native/engine/Tuner.h) sits in every RTEngine and the note travels in the engine's /state (`tuner_hz`, `tuner_confidence`) - the page
reads it from the same /native/monitor/state it already polls. This module only covers the remaining case, nothing else running:
it hands the interface to the native engine opened on the guitar channel alone, with the output muted (-100 dBFS, nothing reaches the
pedal or the amp), and gives it back with stop / when anything else needs the device (live.release_tuner).
"""

from fastapi import APIRouter

from .core import _channel_of, _device_name, _native_call, _opened, fail, session, st
from .live import stop_native
from capture.session import EngineError

router = APIRouter()

MUTED_DB = -100.0


@router.post("/native/tuner/start")
def native_tuner_start():
    if st.tuner_active is not None:
        return {"guitar_ch": st.tuner_active["guitar_ch"]}
    if st.live_active is not None or st.player_active is not None:
        raise fail(409, "srv.cap.tuner_busy")
    opened = _opened()
    guitar_ch, out_ch = opened["in_channels"][0], _channel_of(opened, "out")
    name = _device_name(opened)
    blocksize = (opened.get("info") or {}).get("blocksize", 0)
    st.native_prev = {"device": opened["device"], "in_channels": list(opened["in_channels"]), "out_channels": list(opened["out_channels"]),
                      "blocksize": blocksize}
    session().close_device()
    try:
        _native_call("POST", "/open", {"device": name, "in_channel": guitar_ch, "out_channel": out_ch, "extra_out_channels": [],
                                        "blocksize": blocksize, "gain_db": MUTED_DB, "ceiling_db": -1.0})
    except Exception:
        try:
            session().open(**st.native_prev)
        except EngineError:
            pass
        st.native_prev = None
        raise
    st.tuner_active = {"guitar_ch": guitar_ch}
    return {"guitar_ch": guitar_ch}


@router.get("/native/tuner/active")
def native_tuner_active():
    return {"active": st.tuner_active}


@router.post("/native/tuner/stop")
def native_tuner_stop(restore: bool = True):
    if st.tuner_active is None:
        return {}
    return stop_native(restore)
