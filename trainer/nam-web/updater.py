"""Safe updates of the two trainers (neural-amp-modeler and NAM Trainer Reloaded).

Both trainers live in ../trainers/<name>/ as an unmodified `nam` package plus source.json.
Nothing here touches the venv: updating means replacing one of those folders.

  check    ask for the latest *release* of the upstream project; if it is newer, download it into
           a scratch folder and report what changes (release notes, files, new suspicious code)
  install  (after the user confirmed a check) download again, verify, try the new package with a
           real mini-training on the GPU (selftest.py), back the current folder up to ../Backup,
           swap the folders by renaming and verify; on any error the previous version is put back
  restore  put a backup back (the current version is backed up first, so it is reversible)

Only one operation runs at a time (in a thread; the web page polls status()). install/restore are
"exclusive": the server holds off new jobs meanwhile and refuses to start while jobs are queued
or running. A journal-less recovery (recover(), called at server start) finishes or undoes a swap
that was interrupted by a crash or by "Ferma NAM.bat".
"""

import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import paths
import trainers
from i18n import LocalizedError, msg, render, tr

HERE = Path(__file__).resolve().parent
TRAINERS_DIR = trainers.TRAINERS_DIR
BACKUP_DIR = paths.BACKUP_DIR
SELFTEST = HERE / "selftest.py"
KEEP_BACKUPS = 5

ALLOWED_HOSTS = {"pypi.org", "files.pythonhosted.org", "api.github.com", "github.com", "codeload.github.com"}
MAX_DOWNLOAD = 60 * 1024 * 1024
MAX_UNPACKED = 150 * 1024 * 1024
MAX_FILES = 4000
# File types copied out of an archive; anything else inside nam/ is skipped and reported.
ALLOWED_EXT = {".py", ".pyi", ".json", ".md", ".txt", ".wav", ".yml", ".yaml", ".cfg", ".toml", ".typed", ".rst", ""}
SELFTEST_TIMEOUT_S = 1800

_TAG_RE = re.compile(r"^[A-Za-z0-9._+-]+$")
_BACKUP_ID_RE = re.compile(r"^\d{8}-\d{6}_[A-Za-z0-9._+-]*$")


class UpdateError(LocalizedError):
    """An expected failure. UpdateError("upd.some_key", name=value): the text lives in static/i18n.json and is
    rendered in the language of whoever reads it."""


class Busy(UpdateError):
    pass


# ----------------------------------------------------------------- configuration

_cfg = {"jobs_active": lambda: False, "tmp_dir": paths.TMP_DIR, "cache_dir": paths.CACHE_DIR}


def configure(*, jobs_active: Callable[[], bool], tmp_dir: Path, cache_dir: Path) -> None:
    _cfg.update(jobs_active=jobs_active, tmp_dir=Path(tmp_dir), cache_dir=Path(cache_dir))


# ----------------------------------------------------------------------- state

_lock = threading.RLock()
_state = {"busy": False, "task": None, "name": None, "log": [], "results": {}}
exclusive = False  # read without a lock on purpose: see is_exclusive()


def is_exclusive() -> bool:
    """True while an install/restore is swapping folders: the server must not start jobs."""
    return exclusive


def log(key: str, **params) -> None:
    """Add a line to the operation log. Stored language-neutral ({"t", "k", "p"}) and rendered by status()."""
    entry = {"t": time.strftime("%H:%M:%S"), "k": key, "p": params}
    with _lock:
        _state["log"].append(entry)
        del _state["log"][:-400]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------- network


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _check_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise UpdateError("upd.url_not_allowed", url=url)


def http_get(url: str, max_bytes: int = MAX_DOWNLOAD, timeout: int = 30) -> bytes:
    _check_url(url)
    request = urllib.request.Request(url, headers={
        "User-Agent": "NAM-Trainer-Web-Updater", "Accept": "application/vnd.github+json, application/json, */*"})
    try:
        with urllib.request.build_opener(_SafeRedirect).open(request, timeout=timeout) as response:
            data = response.read(max_bytes + 1)
    except urllib.error.HTTPError as e:
        hint = msg("upd.rate_limit_hint") if e.code in (403, 429) else ""
        raise UpdateError("upd.http_error", code=e.code, host=urllib.parse.urlparse(url).hostname, hint=hint) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise UpdateError("upd.network_unreachable", reason=str(getattr(e, "reason", e))) from e
    if len(data) > max_bytes:
        raise UpdateError("upd.download_too_big")
    return data


_fetch: Callable[..., bytes] = http_get


def set_fetcher(fn: Optional[Callable[..., bytes]]) -> None:
    """Replace the network layer (tests); None restores the real one."""
    global _fetch
    _fetch = fn or http_get


def _fetch_json(url: str):
    try:
        return json.loads(_fetch(url))
    except ValueError as e:
        raise UpdateError("upd.bad_response", host=urllib.parse.urlparse(url).hostname) from e


# --------------------------------------------------------------------- packages


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _strip_pycache(root: Path) -> None:
    for cache in list(root.rglob("__pycache__")):
        _rmtree(cache)


def _extract(data: bytes, dest: Path, *, wheel: bool) -> list:
    """Copy nam/** (allowed file types only) and the LICENSE out of a wheel / GitHub zip into dest.
    Returns the names of files that were skipped because of their type."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise UpdateError("upd.archive_invalid") from e
    infos = archive.infolist()
    if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_UNPACKED:
        raise UpdateError("upd.archive_too_big")
    root = ""
    if not wheel:  # GitHub archives wrap everything in a single "<repo>-<tag>/" folder
        first = infos[0].filename.split("/", 1)[0] + "/" if infos else ""
        if not first or any(not i.filename.startswith(first) for i in infos):
            raise UpdateError("upd.archive_structure")
        root = first
    skipped, dest = [], dest.resolve()
    for info in infos:
        if info.is_dir():
            continue
        rel = info.filename[len(root):]
        posix = PurePosixPath(rel)
        if posix.is_absolute() or ".." in posix.parts or "\\" in rel or any(":" in p for p in posix.parts):
            raise UpdateError("upd.archive_unsafe_path", name=info.filename)
        parts = posix.parts
        if not parts or "__pycache__" in parts:
            continue
        if parts[0] == "nam":
            target = dest.joinpath(*parts)
            if target.suffix.lower() not in ALLOWED_EXT:
                skipped.append(rel)
                continue
        elif len(parts) == 1 and parts[0].upper().startswith(("LICENSE", "COPYING")):
            target = dest / "LICENSE"
        elif wheel and parts[0].endswith(".dist-info") and parts[-1].upper().startswith("LICENSE") and not (dest / "LICENSE").exists():
            target = dest / "LICENSE"
        else:
            continue
        if dest not in target.resolve().parents:
            raise UpdateError("upd.archive_unsafe_path", name=info.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
    if not (dest / "nam" / "__init__.py").exists():
        raise UpdateError("upd.archive_no_nam")
    return skipped


class _OfficialSpec:
    name = "official"
    repo = "sdatkinson/neural-amp-modeler"
    title_key = "upd.title.official"
    button = "Check neural-amp-modeler"

    def latest(self) -> dict:
        data = _fetch_json("https://pypi.org/pypi/neural-amp-modeler/json")
        version = data["info"]["version"]
        wheels = [f for f in data["urls"] if f.get("packagetype") == "bdist_wheel" and f["filename"].endswith("py3-none-any.whl")]
        if not wheels:
            raise UpdateError("upd.no_wheel", version=version)
        wheel = wheels[0]
        _check_url(wheel["url"])
        notes = ""
        if _TAG_RE.match(version):
            try:  # the release notes live on GitHub; not essential
                notes = _fetch_json(f"https://api.github.com/repos/{self.repo}/releases/tags/v{version}").get("body") or ""
            except UpdateError:
                pass
        return {
            "version": version, "tag": f"v{version}", "published": (wheel.get("upload_time_iso_8601") or "")[:10],
            "download_url": wheel["url"], "sha256": wheel["digests"]["sha256"], "notes": notes,
            "requires_python": data["info"].get("requires_python"), "requires_dist": data["info"].get("requires_dist") or [],
        }

    def fetch(self, latest: dict, dest: Path) -> list:
        data = _fetch(latest["download_url"])
        if hashlib.sha256(data).hexdigest() != latest["sha256"]:
            raise UpdateError("upd.sha_mismatch")
        skipped = _extract(data, dest, wheel=True)
        source = {"name": self.name, "repo": self.repo, "kind": "pypi", "version": latest["version"], "tag": latest["tag"],
                  "sha256": latest["sha256"], "released": latest["published"], "installed": now_iso()}
        (dest / "source.json").write_text(json.dumps(source, indent=1), encoding="utf-8")
        return skipped


class _ReloadedSpec:
    name = "reloaded"
    repo = "SlamminMofo/NAM-Trainer-Reloaded"
    title_key = "upd.title.reloaded"
    button = "Check NAM-Trainer-Reloaded"

    def latest(self) -> dict:
        release = _fetch_json(f"https://api.github.com/repos/{self.repo}/releases/latest")
        tag = release["tag_name"]
        if not _TAG_RE.match(tag):
            raise UpdateError("upd.bad_tag", tag=tag)
        commit = _fetch_json(f"https://api.github.com/repos/{self.repo}/commits/{tag}")["sha"]
        return {
            "version": tag.lstrip("vV"), "tag": tag, "commit": commit, "published": (release.get("published_at") or "")[:10],
            "notes": release.get("body") or "", "release_name": release.get("name") or tag,
            "download_url": f"https://github.com/{self.repo}/archive/refs/tags/{tag}.zip",
        }

    def fetch(self, latest: dict, dest: Path) -> list:
        skipped = _extract(_fetch(latest["download_url"]), dest, wheel=False)
        source = {"name": self.name, "repo": self.repo, "kind": "github", "version": latest["version"], "tag": latest["tag"],
                  "commit": latest["commit"], "released": latest["published"], "installed": now_iso()}
        (dest / "source.json").write_text(json.dumps(source, indent=1), encoding="utf-8")
        return skipped


SPECS = {"official": _OfficialSpec(), "reloaded": _ReloadedSpec()}


# -------------------------------------------------------------- comparing releases


def compare_release(latest: dict, installed: dict) -> str:
    """'newer' | 'same' | 'older' (relative to what is installed)."""
    have = installed.get("version")
    if not have:
        return "newer"
    try:
        from packaging.version import Version

        a, b = Version(latest["version"]), Version(have)
    except Exception:  # noqa: BLE001 - odd version strings: fall back to plain inequality
        return "same" if latest["version"] == have else "newer"
    if a > b:
        return "newer"
    if a < b:
        return "older"
    installed_commit, latest_commit = installed.get("commit"), latest.get("commit")
    if installed_commit and latest_commit and installed_commit != latest_commit:
        return "newer"  # same tag, different commit: the release was re-published
    return "same"


def check_requirements(latest: dict) -> list:
    """Requirements of a new official release that the venv does not satisfy."""
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    unmet = []
    python_req = latest.get("requires_python")
    running = Version(".".join(map(str, sys.version_info[:3])))
    if python_req and not SpecifierSet(python_req).contains(running, prereleases=True):
        unmet.append(msg("upd.req_python", need=python_req, have=str(running)))
    for text in latest.get("requires_dist") or []:
        try:
            req = Requirement(text)
        except Exception:  # noqa: BLE001
            continue
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue  # optional extra, or another platform
        try:
            installed = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            unmet.append(msg("upd.req_missing", name=req.name, spec=str(req.specifier) or msg("upd.any_version")))
            continue
        if req.specifier and not req.specifier.contains(Version(installed), prereleases=True):
            unmet.append(msg("upd.req_version", name=req.name, spec=str(req.specifier), have=installed))
    return unmet


# ------------------------------------------------------------------- change scan

_PATTERNS = [
    ("flag.network", re.compile(r"^\s*(import|from)\s+(requests|urllib|http|socket|ftplib|smtplib|websockets?|aiohttp|httpx|xmlrpc|telnetlib|paramiko)\b")),
    ("flag.commands", re.compile(r"\b(subprocess|os\.system|os\.popen|os\.exec\w*|os\.spawn\w*|pty\.spawn)\b")),
    ("flag.dynamic", re.compile(r"(?<![\w.])(exec|eval|compile)\s*\(|__import__\s*\(")),
    ("flag.obfuscation", re.compile(r"\b(base64|marshal|pickle|dill|cloudpickle|zlib\.decompress|codecs\.decode)\b")),
    ("flag.system", re.compile(r"\b(ctypes|winreg|_winapi|win32api|win32com|msvcrt)\b")),
    ("flag.delete", re.compile(r"\b(shutil\.rmtree|os\.remove|os\.unlink|os\.rmdir)\b|\.unlink\(")),
]
_TEXT_EXT = {".py", ".json", ".md", ".txt", ".yml", ".yaml", ".cfg", ".toml", ".rst"}
_LIST_CAP = 200


def _files_of(root: Optional[Path]) -> dict:
    """The package files under <root>/nam (paths are relative to root, so they start with 'nam/')."""
    if root is None or not (root / "nam").exists():
        return {}
    return {p.relative_to(root).as_posix(): p for p in sorted((root / "nam").rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def _lines(path: Path) -> list:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def scan_packages(old_dir: Optional[Path], new_dir: Path, skipped: list = ()) -> dict:
    """What changes between the installed package and a new one, and which *new* lines look risky
    (network, subprocess, dynamic code...). Patterns are only a hint: not a security guarantee."""
    old, new = _files_of(old_dir), _files_of(new_dir)
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    modified, unchanged, flags = [], 0, []
    for rel in sorted(set(new) & set(old)):
        a, b = old[rel].read_bytes(), new[rel].read_bytes()
        if a == b:
            unchanged += 1
            continue
        entry = {"file": rel, "plus": None, "minus": None}
        if new[rel].suffix.lower() in _TEXT_EXT:
            before = Counter(l.strip() for l in _lines(old[rel]) if l.strip())
            after = Counter(l.strip() for l in _lines(new[rel]) if l.strip())
            entry["plus"] = sum((after - before).values())
            entry["minus"] = sum((before - after).values())
        modified.append(entry)
    for rel in added + [m["file"] for m in modified]:
        if not rel.endswith(".py"):
            continue
        known = {l.strip() for l in _lines(old[rel])} if rel in old else set()
        for number, text in enumerate(_lines(new[rel]), 1):
            line = text.strip()
            if not line or line.startswith("#") or line in known:
                continue
            for kind, pattern in _PATTERNS:
                if pattern.search(line):
                    flags.append({"file": rel, "line": number, "kind": kind, "text": line[:160]})
                    break
    for rel in skipped:
        flags.append({"file": rel, "line": 0, "kind": "flag.disallowed_file", "text": ""})
    return {
        "added": added[:_LIST_CAP], "removed": removed[:_LIST_CAP], "modified": modified[:_LIST_CAP], "unchanged": unchanged,
        "counts": {"added": len(added), "removed": len(removed), "modified": len(modified)},
        "flags": flags[:_LIST_CAP], "flag_count": len(flags),
    }


# --------------------------------------------------------------- backups / swap


def _installed_dir(name: str) -> Path:
    return TRAINERS_DIR / name


def list_backups(name: str) -> list:
    root = BACKUP_DIR / name
    out = []
    if root.exists():
        for d in sorted((p for p in root.iterdir() if p.is_dir() and _BACKUP_ID_RE.match(p.name)), reverse=True):
            try:
                info = json.loads((d / "source.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                info = {}
            out.append({"id": d.name, "version": info.get("version", "?"), "tag": info.get("tag"),
                        "commit": info.get("commit"), "installed": info.get("installed")})
    return out


def _new_backup_dir(name: str, version: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._+-]", "_", version or "sconosciuta")
    root = BACKUP_DIR / name
    root.mkdir(parents=True, exist_ok=True)
    base = time.strftime("%Y%m%d-%H%M%S")
    candidate, i = root / f"{base}_{safe}", 1
    while candidate.exists():  # two backups within the same second
        i += 1
        candidate = root / f"{base}-{i}_{safe}"
    return candidate


def _prune_backups(name: str) -> None:
    for entry in list_backups(name)[KEEP_BACKUPS:]:
        _rmtree(BACKUP_DIR / name / entry["id"])


def _backup_current(name: str) -> Optional[Path]:
    current = _installed_dir(name)
    if not (current / "nam" / "__init__.py").exists():
        return None
    target = _new_backup_dir(name, trainers.installed_info(name).get("version", ""))
    shutil.copytree(current, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def _rename_retry(src: Path, dst: Path, tries: int = 6, delay: float = 1.0) -> None:
    for attempt in range(tries):
        try:
            os.rename(src, dst)
            return
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(delay)  # antivirus / indexer briefly holding a file


def _swap_in(name: str, new_dir: Path) -> Path:
    """current -> .old-<name>, new_dir -> current. Returns the .old folder (kept until verified)."""
    current, old = _installed_dir(name), TRAINERS_DIR / f".old-{name}"
    _rmtree(old)
    if current.exists():
        _rename_retry(current, old)
    try:
        _rename_retry(new_dir, current)
    except OSError:
        if old.exists():
            _rename_retry(old, current)
        raise
    return old


def _rollback_swap(name: str, old: Path) -> None:
    current = _installed_dir(name)
    if old.exists():
        _rmtree(current)
        _rename_retry(old, current)


def _child_env() -> dict:
    tmp, cache = str(_cfg["tmp_dir"]), _cfg["cache_dir"]
    Path(tmp).mkdir(parents=True, exist_ok=True)
    return {**os.environ, "PYTHONUTF8": "1", "TEMP": tmp, "TMP": tmp, "MPLCONFIGDIR": str(cache / "matplotlib"), "MPLBACKEND": "Agg"}


def _verify_import(name: str, expect_version: str) -> None:
    """Import the installed package in a fresh interpreter: it must import, and come from the installed folder.
    A different declared version is only noted (some projects don't bump it inside the package)."""
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import nam; print('VERSION', nam.__version__, nam.__file__)")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    installed = _installed_dir(name).resolve()
    proc = subprocess.run([sys.executable, "-c", code, str(installed)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300, env=_child_env(), creationflags=flags)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("VERSION ")), "")
    parts = line.split(" ", 2)
    if proc.returncode != 0 or len(parts) < 3:
        raise UpdateError("upd.import_failed", detail=(proc.stderr.strip().splitlines() or ["?"])[-1])
    if Path(parts[2]).resolve().parent != installed / "nam":
        raise UpdateError("upd.wrong_path", path=parts[2])
    if parts[1] != expect_version:
        log("upd.note_declared_version", declared=parts[1], expected=expect_version)


def recover() -> list:
    """At server start: finish or undo an interrupted swap and remove scratch folders."""
    notes = []
    for name in SPECS:
        current, old = _installed_dir(name), TRAINERS_DIR / f".old-{name}"
        if old.exists():
            if (current / "nam" / "__init__.py").exists():
                _rmtree(old)  # the swap had completed, only the clean-up was left
                notes.append(f"{name}: pulizia di un aggiornamento gia completato")
            else:
                _rmtree(current)
                _rename_retry(old, current)
                notes.append(f"{name}: ripristinata la versione precedente dopo un aggiornamento interrotto")
        _rmtree(TRAINERS_DIR / f".staging-{name}")
    return notes


# ----------------------------------------------------------------------- selftest


def _run_selftest(name: str, package: Path, expect_version: str, emit_catalog: Optional[Path]) -> None:
    workdir = Path(_cfg["tmp_dir"]) / f"selftest-{name}-{int(time.time())}"
    command = [sys.executable, "-u", str(SELFTEST), "run", "--trainer", name, "--path", str(package),
               "--workdir", str(workdir), "--expect-version", expect_version]
    if emit_catalog:
        command += ["--emit-catalog", str(emit_catalog)]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=_child_env(), creationflags=flags, stdin=subprocess.DEVNULL)
    watchdog = threading.Timer(SELFTEST_TIMEOUT_S, proc.kill)
    watchdog.start()
    result, tail = None, []
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("RESULT "):
                try:
                    result = json.loads(line[len("RESULT "):])
                except ValueError:
                    pass
            elif line.startswith("[selftest]"):
                # "[selftest] sel.some_key {json params}" (see selftest.say); anything else is shown as it is
                key, _, payload = line[len("[selftest] "):].partition(" ")
                try:
                    log(key, **(json.loads(payload) if payload else {})) if key.startswith("sel.") else log("upd.raw", text=line[len("[selftest] "):])
                except (ValueError, TypeError):
                    log("upd.raw", text=line[len("[selftest] "):])
            elif line:
                tail.append(line)
        proc.wait()
    finally:
        watchdog.cancel()
        _rmtree(workdir)
    if result is None:
        raise UpdateError("upd.selftest_no_result", tail=" | ".join(tail[-4:]))
    if not result.get("ok"):
        raise UpdateError("upd.selftest_failed", reason=result.get("error") or msg("upd.unknown_reason"))


# -------------------------------------------------------------------- operations


def _require_idle_jobs() -> None:
    if _cfg["jobs_active"]():
        raise UpdateError("upd.jobs_active")


def do_check(name: str) -> dict:
    spec = SPECS[name]
    installed = trainers.installed_info(name)
    log("upd.check_searching", repo=spec.repo)
    latest = spec.latest()
    verdict = compare_release(latest, installed)
    result = {
        "status": "up_to_date", "checked_at": now_iso(),
        "installed": {k: installed.get(k) for k in ("version", "tag", "commit", "released", "installed")},
        "latest": {"version": latest["version"], "tag": latest["tag"], "published": latest["published"],
                   "commit": latest.get("commit")},
    }
    if verdict == "same":
        log("upd.check_up_to_date", version=latest["version"])
        return result
    if verdict == "older":
        result["note"] = msg("upd.installed_newer_note")
        log("upd.installed_newer_note")
        return result
    log("upd.check_found", latest=latest["version"], installed=installed.get("version") or msg("upd.unknown_version"))
    scratch = Path(_cfg["tmp_dir"]) / f"update-check-{name}"
    _rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        log("upd.check_downloading")
        skipped = spec.fetch(latest, scratch)
        unmet = check_requirements(latest) if name == "official" else []
        report = scan_packages(_installed_dir(name), scratch, skipped)
    finally:
        _rmtree(scratch)
    result.update(status="update_available", notes=latest["notes"][:6000], report=report, unmet=unmet)
    log("upd.check_summary", modified=report["counts"]["modified"], added=report["counts"]["added"],
        removed=report["counts"]["removed"], flags=report["flag_count"])
    return result


def do_install(name: str, target: str) -> dict:
    spec = SPECS[name]
    installed = trainers.installed_info(name)
    log("upd.install_recheck")
    latest = spec.latest()
    if latest["version"] != target:
        raise UpdateError("upd.release_changed", target=target, latest=latest["version"])
    staging = TRAINERS_DIR / f".staging-{name}"
    _rmtree(staging)
    staging.mkdir(parents=True)
    try:
        log("upd.install_downloading")
        skipped = spec.fetch(latest, staging)
        if name == "official":
            unmet = check_requirements(latest)
            if unmet:
                raise UpdateError("upd.unmet_requirements", items=unmet)
        report = scan_packages(_installed_dir(name), staging, skipped)
        log("upd.install_summary", modified=report["counts"]["modified"], added=report["counts"]["added"],
            removed=report["counts"]["removed"], flags=report["flag_count"])
        log("upd.install_selftest")
        _run_selftest(name, staging, latest["version"], staging / "catalog.json" if name == "reloaded" else None)
        _strip_pycache(staging)
    except BaseException:
        _rmtree(staging)
        raise
    backup = _backup_current(name)
    if backup:
        log("upd.backup_made", name=name, backup=backup.name)
    log("upd.applying")
    old = _swap_in(name, staging)
    try:
        _verify_import(name, latest["version"])
    except BaseException as e:
        log("upd.verify_failed_rollback")
        _rollback_swap(name, old)
        raise UpdateError("upd.update_cancelled", reason=_reason(e)) from e
    _rmtree(old)
    _prune_backups(name)
    log("upd.update_done", old=installed.get("version") or msg("upd.unknown_version"), new=latest["version"])
    return {"status": "installed", "from": installed.get("version"), "version": latest["version"],
            "backup": backup.name if backup else None}


def do_restore(name: str, backup_id: str) -> dict:
    if not _BACKUP_ID_RE.match(backup_id or ""):
        raise UpdateError("upd.bad_backup")
    source = BACKUP_DIR / name / backup_id
    if not (source / "nam" / "__init__.py").exists():
        raise UpdateError("upd.backup_missing")
    try:
        target_version = json.loads((source / "source.json").read_text(encoding="utf-8"))["version"]
    except (OSError, ValueError, KeyError):
        raise UpdateError("upd.backup_no_version")
    current_version = trainers.installed_info(name).get("version")
    staging = TRAINERS_DIR / f".staging-{name}"
    _rmtree(staging)
    log("upd.restoring", version=target_version, backup=backup_id)
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns("__pycache__"))
    backup = _backup_current(name)
    if backup:
        log("upd.backup_current", version=current_version, name=name, backup=backup.name)
    old = _swap_in(name, staging)
    try:
        _verify_import(name, target_version)
    except BaseException as e:
        log("upd.restore_verify_failed")
        _rollback_swap(name, old)
        raise UpdateError("upd.restore_cancelled", reason=_reason(e)) from e
    _rmtree(old)
    _prune_backups(name)
    log("upd.restore_done", old=current_version, new=target_version)
    return {"status": "restored", "from": current_version, "version": target_version}


# ------------------------------------------------------------------ public API


def _reason(exc: BaseException):
    """Language-neutral form of an exception, to be embedded in another message."""
    return exc.message() if isinstance(exc, LocalizedError) else str(exc)


def _start(kind: str, name: str, fn: Callable[[], dict]) -> None:
    global exclusive
    with _lock:
        if _state["busy"]:
            raise Busy("upd.busy")
        _state.update(busy=True, task=kind, name=name, log=[])
        exclusive = kind in ("install", "restore")

    def runner():
        global exclusive
        try:
            result = fn()
        except UpdateError as e:
            log("upd.log_error", detail=e.message())
            result = {"status": "error", "error": e.message()}
        except Exception as e:  # noqa: BLE001 - never let a bug kill the thread silently
            log("upd.log_unexpected", detail=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            result = {"status": "error", "error": msg("upd.error_raw", detail=f"{type(e).__name__}: {e}")}
        with _lock:
            slot = _state["results"].setdefault(name, {})
            if kind == "check":
                slot["check"] = result
            else:
                slot["last_op"] = {**result, "kind": kind, "at": now_iso()}
                if result.get("status") in ("installed", "restored"):
                    slot.pop("check", None)  # what it said is stale now
            _state["busy"] = False
            exclusive = False

    threading.Thread(target=runner, name=f"updater-{kind}-{name}", daemon=True).start()


def _spec(name: str):
    if name not in SPECS:
        raise UpdateError("upd.unknown_trainer", name=name)
    return SPECS[name]


def start_check(name: str) -> None:
    _spec(name)
    _start("check", name, lambda: do_check(name))


def start_install(name: str, target: str) -> None:
    _spec(name)
    _require_idle_jobs()
    _start("install", name, lambda: do_install(name, target))


def start_restore(name: str, backup_id: str) -> None:
    _spec(name)
    _require_idle_jobs()
    _start("restore", name, lambda: do_restore(name, backup_id))


def _rendered(result: Optional[dict], lang: Optional[str]) -> Optional[dict]:
    """A copy of a stored check/operation result with its language-neutral messages turned into text."""
    if result is None:
        return None
    out = dict(result)
    for key in ("error", "note"):
        if key in out:
            out[key] = render(out[key], lang)
    if "unmet" in out:
        out["unmet"] = [render(item, lang) for item in out["unmet"]]
    return out


def status(lang: Optional[str] = None) -> dict:
    """Everything the page shows about updates, with every message rendered in `lang` (default: the request's)."""
    jobs_active = bool(_cfg["jobs_active"]())
    with _lock:
        lines = [f"{e['t']} {tr(e['k'], lang, **e['p'])}" for e in _state["log"]]
        out = {"busy": _state["busy"], "task": _state["task"], "task_name": _state["name"], "exclusive": exclusive,
               "log": lines, "jobs_active": jobs_active, "trainers": {}}
        results = {k: dict(v) for k, v in _state["results"].items()}
    for name, spec in SPECS.items():
        info = trainers.installed_info(name)
        out["trainers"][name] = {
            "title": tr(spec.title_key, lang), "button": spec.button, "repo": spec.repo,
            "installed": {k: info.get(k) for k in ("version", "tag", "commit", "released", "installed")},
            "backups": list_backups(name), "check": _rendered(results.get(name, {}).get("check"), lang),
            "last_op": _rendered(results.get(name, {}).get("last_op"), lang),
        }
    return out
