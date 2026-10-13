"""Hand-off to the trainer (nam-trainer-gui, a separate program on the Windows PC): detected live on the LAN by a
small HTTP probe, no shared folder (Roman: "non voglio cartelle condivise, voglio una luce... se rileva il NAM
Trainer in rete"). The page polls GET /trainer/status for the light; when it comes back online, "Invia al NAM
Trainer" sends the capture as one JSON POST (the WAV bytes travel base64-encoded inside it - simplest possible
contract, no multipart parsing needed on the trainer's side).

nam-trainer-gui exposes these two routes on TRAINER_PORT (same port its own web GUI already
listens on - nam-web/start.ps1 hardcodes 8765 - these two routes are meant to live on that same FastAPI app, not a
second server):
    GET  /nam-trainer/ping     -> any 2xx response means "I'm here"
    POST /nam-trainer/import   -> JSON body {name, input_rel, input_md5, input_size, metadata, report, wav_name, wav_base64};
                                  answers {received, input_rel, input_needed}
    POST /nam-trainer/import-input -> only if input_needed: JSON body {capture_rel (= received), name, md5, wav_base64}
The trainer asks for the input.wav of a capture only when it has no file with that md5 yet, so it travels once, not with every capture.
"""

import base64
import hashlib
import http.client
import json
import re
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from fastapi import APIRouter
from pydantic import BaseModel

import settings

from .core import _ctx, fail
from .live import _live_dir, _live_tmp_dir

router = APIRouter()

TRAINER_PORT = 8765   # nam-web's own web GUI port (start.ps1), not a separate service
HEALTH_PATH = "/nam-trainer/ping"
IMPORT_PATH = "/nam-trainer/import"
IMPORT_INPUT_PATH = "/nam-trainer/import-input"
PING_TIMEOUT = 1.2
SEND_TIMEOUT = 60.0
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def _trainer_request(host: str, method: str, path: str, body: Optional[dict], timeout: float) -> Optional[int]:
    """None on any connection problem (host off, wrong address, firewall) - never raises, callers turn that into
    "not online" or a clear refusal."""
    conn = http.client.HTTPConnection(host, TRAINER_PORT, timeout=timeout)
    try:
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        conn.request(method, path, body=payload, headers=headers)
        status = conn.getresponse().status
        return status
    except (OSError, ValueError):
        return None
    finally:
        conn.close()


def _trainer_post_json(host: str, path: str, body: dict, timeout: float):
    """(status, answer as a dict - empty if the trainer sent no JSON) of a POST, or (None, {}) on any connection problem."""
    conn = http.client.HTTPConnection(host, TRAINER_PORT, timeout=timeout)
    try:
        conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read()
        try:
            answer = json.loads(raw) if raw else {}
        except ValueError:
            answer = {}
        return response.status, answer if isinstance(answer, dict) else {}
    except (OSError, ValueError):
        return None, {}
    finally:
        conn.close()


def clean_host(value: str) -> str:
    """What a person types for "the address of the trainer" is not always an address: the trainer's page used to show
    "http://192.168.1.20:8765", and that is what ends up pasted. Keeps the host only (no scheme, port, path); "" if nothing usable is left."""
    v = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", value.strip())
    v = re.split(r"[/?#]", v, maxsplit=1)[0].strip()
    if v.startswith("[") and "]" in v:
        v = v[1:v.index("]")]
    elif v.count(":") == 1:
        v = v.split(":")[0]
    v = v.rstrip(".")
    return v if re.fullmatch(r"[A-Za-z0-9._-]+|[0-9A-Fa-f:]+", v or "") else ""


def _is_online(host: str) -> bool:
    if not host:
        return False
    status = _trainer_request(host, "GET", HEALTH_PATH, None, PING_TIMEOUT)
    return status is not None and 200 <= status < 300


@router.get("/trainer/status")
def trainer_status():
    host = settings.load()["trainer_host"]
    online = _is_online(host)
    # a wrong address is the commonest reason for "not found": say if a trainer answers on this very PC
    local = online if host in LOCAL_HOSTS else (not online and _is_online("localhost"))
    # where the trainer's own page is (the same address the probe uses): the page shows it as a link to open it in the browser
    return {"host": host or None, "online": online, "local_online": local, "url": f"http://{host}:{TRAINER_PORT}/" if host else None}


class TrainerHostBody(BaseModel):
    host: str


@router.post("/trainer/host")
def trainer_host_set(body: TrainerHostBody):
    host = clean_host(body.host)
    if not host:
        raise fail(400, "srv.cap.trainer_bad_host")
    settings.update(trainer_host=host)
    return {"host": host, "online": _is_online(host)}


def _read_capture_json(d: Path) -> Optional[dict]:
    try:
        return json.loads((d / "capture.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _capture_entry(d: Path) -> Optional[dict]:
    """None if `d` isn't a capture folder at all; otherwise a summary, `sendable` false when the aligned WAV/JSON
    the trainer needs aren't there yet (e.g. still recording, or discarded) - listed anyway, not hidden, so browsing
    an old library folder shows why a given capture can't be sent."""
    rec = _read_capture_json(d)
    if rec is None:
        return None
    output = (rec.get("files") or {}).get("output")
    sendable = False
    if output:
        wav = d / Path(output).name
        sendable = wav.is_file() and wav.with_suffix(".json").is_file()
    return {"dir": str(d), "name": rec.get("name"), "status": rec.get("status"), "created": rec.get("created"),
            "exportable": sendable}


class TrainerScanBody(BaseModel):
    path: str


@router.post("/trainer/scan")
def trainer_scan(body: TrainerScanBody):
    """`path` comes from a native folder panel (packaging/main.swift), so it can be anywhere - not just this
    library. Looks at `path` itself (one capture folder picked directly) and, failing that, one level under it
    (a "Catture"-like folder picked instead, holding many capture folders)."""
    root = Path(body.path)
    if not root.is_dir():
        raise fail(400, "srv.cap.trainer_bad_folder")
    self_entry = _capture_entry(root)
    if self_entry is not None:
        return {"captures": [self_entry]}
    found: List[dict] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            entry = _capture_entry(child)
            if entry is not None:
                found.append(entry)
    return {"captures": found}


class TrainerSendBody(BaseModel):
    dir: Optional[str] = None          # an absolute folder, from /trainer/scan's own results
    folder_rel: Optional[str] = None   # or a path relative to this library, from a capture just accepted


def _resolve_send_dir(body: TrainerSendBody) -> Path:
    if body.folder_rel:
        wav_dir = _ctx["wav_dir"].resolve()
        d = (wav_dir / body.folder_rel).resolve()
        if wav_dir != d and wav_dir not in d.parents:
            raise fail(400, "srv.cap.trainer_bad_folder")
        return d
    if body.dir:
        return Path(body.dir)
    raise fail(400, "srv.cap.trainer_bad_folder")


@router.post("/trainer/send")
def trainer_send(body: TrainerSendBody):
    d = _resolve_send_dir(body)
    entry = _capture_entry(d) if d.is_dir() else None
    if entry is None or not entry["exportable"]:
        raise fail(409, "srv.cap.trainer_no_files")
    host = settings.load()["trainer_host"]
    if not host:
        raise fail(409, "srv.cap.trainer_not_configured")
    rec = _read_capture_json(d) or {}
    wav_name = Path((rec.get("files") or {}).get("output")).name
    json_name = Path(wav_name).with_suffix(".json").name
    try:
        sidecar = json.loads((d / json_name).read_text(encoding="utf-8"))
        wav_b64 = base64.b64encode((d / wav_name).read_bytes()).decode("ascii")
    except OSError as e:
        raise fail(500, "srv.cap.write_failed", detail=str(e))
    input_file = _input_file_of(rec, sidecar)
    payload = {**sidecar, "wav_name": wav_name, "wav_base64": wav_b64}
    if input_file is not None:
        payload.update(input_md5=_md5(input_file), input_size=input_file.stat().st_size)
    status, answer = _trainer_post_json(host, IMPORT_PATH, payload, SEND_TIMEOUT)
    if status is None or status >= 300:
        raise fail(503, "srv.cap.trainer_unreachable")
    sent_input = False
    if input_file is not None and answer.get("input_needed") and answer.get("received"):
        status, _ = _trainer_post_json(host, IMPORT_INPUT_PATH, {"capture_rel": answer["received"], "name": input_file.name, "md5": payload["input_md5"],
                                                                 "wav_base64": base64.b64encode(input_file.read_bytes()).decode("ascii")}, SEND_TIMEOUT)
        if status is None or status >= 300:
            raise fail(503, "srv.cap.trainer_input_failed")
        sent_input = True
    return {"sent": wav_name, "input_sent": sent_input}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _input_file_of(rec: dict, sidecar: dict) -> Optional[Path]:
    """The input.wav this capture was made with, from the library (`input.ref` of capture.json, else the sidecar's `input_rel`);
    None if it is not there any more - the capture is sent anyway, the trainer's form then asks for an input."""
    ref = (rec.get("input") or {}).get("ref") or sidecar.get("input_rel")
    if not ref:
        return None
    wav_dir = _ctx["wav_dir"].resolve()
    path = (wav_dir / ref).resolve()
    return path if wav_dir in path.parents and path.is_file() else None


# ------------------------------------------------------------------------------ the way back: models from the trainer
# capture -> trainer -> back here with the .nam: the NAM player picks a finished model straight from the trainer's own list
# (GET /api/models, GET /api/models/<id>/model: routes its web page already uses), no file to look for in a folder.

MODEL_LIST_PATH = "/api/models"
MODEL_FETCH_TIMEOUT = 60.0


def _trainer_fetch(host: str, path: str, timeout: float):
    """(status, body bytes) of a GET to the trainer, or None on any connection problem (never raises)."""
    conn = http.client.HTTPConnection(host, TRAINER_PORT, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    except (OSError, ValueError):
        return None
    finally:
        conn.close()


@router.get("/trainer/models")
def trainer_models():
    """The finished models the trainer has, newest first - for the player's "Dal Trainer" list. Offline or unconfigured is not an error:
    the page just shows the list empty with the lamp off."""
    host = settings.load()["trainer_host"]
    if not host:
        return {"host": None, "online": False, "models": []}
    got = _trainer_fetch(host, MODEL_LIST_PATH, PING_TIMEOUT * 4)
    if got is None:
        return {"host": host, "online": False, "models": []}
    try:
        entries = json.loads(got[1]) if got[0] == 200 else []
    except ValueError:
        entries = []
    models = [{"id": e["id"], "name": e.get("name") or e["id"], "created": e.get("created"), "size": e.get("model_size"),
               "folder": e.get("folder"), "state": e.get("state")}
              for e in entries if isinstance(e, dict) and e.get("id") and e.get("has_model")]
    models.sort(key=lambda m: m["created"] or 0, reverse=True)
    return {"host": host, "online": True, "models": models[:100]}


class TrainerFetchBody(BaseModel):
    id: str
    name: str = "modello"
    do_import: bool = False       # True: kept in the app's own "Modelli NAM" like any imported file; False: a one-off copy, same as "Sfoglia"


@router.post("/trainer/models/fetch")
def trainer_models_fetch(body: TrainerFetchBody):
    host = settings.load()["trainer_host"]
    if not host:
        raise fail(409, "srv.cap.trainer_not_configured")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", body.id):
        raise fail(400, "srv.cap.bad_filename")
    got = _trainer_fetch(host, f"{MODEL_LIST_PATH}/{quote(body.id, safe='')}/model", MODEL_FETCH_TIMEOUT)
    if got is None:
        raise fail(503, "srv.cap.trainer_unreachable")
    status, data = got
    if status == 404:
        raise fail(404, "srv.cap.trainer_model_missing")
    if status != 200 or not data.lstrip().startswith(b"{"):          # a .nam file is JSON: anything else is not a model
        raise fail(502, "srv.cap.trainer_bad_model")
    if body.do_import:
        stem = _ctx["safe_name"](body.name, fallback="modello")
        dest = _ctx["unique_file"](_live_dir("model") / f"{stem}.nam")
        dest.write_bytes(data)
        return {"ref": dest.name, "imported": True}
    dest = _live_tmp_dir() / f"{uuid.uuid4().hex}.nam"
    dest.write_bytes(data)
    return {"ref": f"tmp:{dest.name}", "imported": False}
