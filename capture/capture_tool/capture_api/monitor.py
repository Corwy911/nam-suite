"""The native engine's generic state proxy - shared by every tool that runs on it (today: "Prova dal vivo")."""

from fastapi import APIRouter, HTTPException

from .core import _native_call

router = APIRouter()


@router.get("/native/monitor/state")
def native_monitor_state():
    try:
        return _native_call("GET", "/state")
    except HTTPException:
        return {"opened": False, "in_peak_dbfs": None, "out_peak_dbfs": None, "xruns": 0}
