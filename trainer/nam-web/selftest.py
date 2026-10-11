"""Try a trainer package (a folder holding `nam/`) before it is installed.

  python selftest.py run --trainer official|reloaded --path <dir> --workdir <dir>
                         --expect-version <v> [--emit-catalog <file>]

`run` proves that the package still works with worker.py, exactly as production uses it:
  1. a child process imports the package and checks that every argument the worker passes to
     `nam.train.core.train` still exists, that the loaded version/path are the expected ones,
     and (Reloaded) builds the architecture catalogue from the package itself;
  2. the real `worker.py` is started (like the server does) on synthetic audio for one or two
     short trainings on the GPU; the model it exports must be valid.
Progress goes to stdout as "[selftest] ..." lines, the outcome as a last line "RESULT {json}".
The synthetic audio is not an official NAM input, so the workers get selftest_hook on
PYTHONPATH (see there); nothing of this touches production workers.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hardware  # noqa: E402  (stdlib only)
HOOK_DIR = HERE / "selftest_hook"
CASE_TIMEOUT_S = 900


def say(key: str, **params) -> None:
    """A progress line for the updater log: "[selftest] <key> <json params>". The texts live in static/i18n.json."""
    print(f"[selftest] {key} {json.dumps(params, ensure_ascii=False)}", flush=True)


def M(key: str, **params) -> dict:
    """A language-neutral message (same shape as i18n.msg), rendered later in the reader's language."""
    return {"k": key, "p": params} if params else {"k": key}


def emit_result(data: dict) -> None:
    print("RESULT " + json.dumps(data), flush=True)


# ----------------------------------------------------------------- introspection


def build_catalog(core, trainers) -> dict:
    """Architecture catalogue of a Reloaded package (same shape as trainers.py's static one)."""
    members, seen = [], set()
    label_of, recommend = (lambda a: a.value), None
    order = []
    try:
        from nam.train import gui  # the Reloaded desktop GUI module: list order, labels, recommended LR

        order = list(gui._ARCHITECTURE_ORDER)
        label_of = gui._architecture_display_label
        recommend = gui._recommended_lr_decay_for_scheduler
    except Exception as e:  # noqa: BLE001 - any failure just means "use the plain enum"
        say("sel.gui_unavailable", error=type(e).__name__)
    for arch in order + list(core.Architecture):
        if arch.value not in seen:
            seen.add(arch.value)
            members.append(arch)

    light = {"standard", "lite", "feather", "nano", "Nano64x4", "Nano125x3"}
    is_lstm = getattr(core, "_is_lstm_only_architecture", lambda a: False)
    is_cc_lstm = getattr(core, "_is_causal_conv_lstm_architecture", lambda a: False)

    def group_of(arch) -> str:
        if arch.value.startswith("A2 "):
            return "a2"
        if arch.value in light:
            return "light"
        return "lstm" if (is_lstm(arch) or is_cc_lstm(arch)) else "extended"

    entries = []
    for arch in members:
        lr, decay = 0.002, 0.004
        if recommend is not None:
            try:
                lr, decay = recommend(core.LearningRateScheduler.EXPONENTIAL, 300, architecture=arch)
            except Exception:  # noqa: BLE001
                pass
        entries.append({"value": arch.value, "label": str(label_of(arch)), "group": group_of(arch),
                        "lr": round(float(lr), 5), "lr_decay": round(float(decay), 5)})

    def enum_values(name, fallback, drop=()):
        enum = getattr(core, name, None)
        return [e.value for e in enum if e.value not in drop] if enum else fallback

    catalog = {
        "entries": entries,
        "schedulers": enum_values("LearningRateScheduler", trainers._STATIC_SCHEDULERS),
        "stage_modes": enum_values("TrainingStageMode", trainers._STATIC_STAGE_MODES, drop=("refinement_only",)),
        "stage2_focus": enum_values("StageTwoFocus", trainers._STATIC_STAGE2_FOCUS),
    }
    if not trainers._valid_catalog(catalog):
        raise RuntimeError("the architecture catalogue generated from the package is not valid")
    return catalog


def introspect(args) -> int:
    """Child process: import the package under test and report on it."""
    import inspect

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(Path(args.path).resolve()))
    import nam
    from nam.models.metadata import UserMetadata  # noqa: F401 - the worker imports it
    from nam.train import core
    import trainers
    import worker

    problems = []
    params = inspect.signature(core.train).parameters
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    job = {"backend": args.trainer, "epochs": 1, "batch_size": 1}
    if args.trainer == "reloaded":
        job.update(architecture="A2 Full+Lite", lr=0.1, lr_decay=0.1, lr_scheduler="exponential", fit_mrstft=True,
                   stage_mode="two_stage", stage2_epochs=1, stage2_lr=0.1, stage2_lr_decay=0.1,
                   stage2_lr_scheduler="exponential", stage2_focus="Lows")
    missing = [] if accepts_any else sorted(set(worker.train_kwargs(job, None)) - set(params))
    if missing:
        problems.append(M("sel.p.train_args", names=", ".join(missing)))
    for attr in ("get_callbacks", "_detect_input_version", "_Version"):
        if not hasattr(core, attr):
            problems.append(M("sel.p.missing_attr", attr=attr))

    catalog = None
    if args.trainer == "reloaded":
        try:
            catalog = build_catalog(core, trainers)
        except Exception as e:  # noqa: BLE001
            problems.append(M("sel.p.catalog", detail=f"{type(e).__name__}: {e}"))
    emit_result({"version": nam.__version__, "file": str(Path(nam.__file__).resolve()), "problems": problems,
                 "catalog": catalog})
    return 0


# ------------------------------------------------------------------ synthetic data


def make_wavs(directory: Path, seconds: int = 30) -> None:
    """A DI-like signal (30 s, or 20 s on a CPU-only install) and its distorted version, laid out like an old
    NAM input (silence before the validation segment, which is the last 9 s)."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import butter, lfilter

    sr = 48000
    n = sr * seconds
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    b, a = butter(2, 4000 / (sr / 2))
    x = lfilter(b, a, x)
    x = np.clip(0.9 * x / np.abs(x).max(), -0.95, 0.95).astype(np.float32)
    x[n - 9 * sr - sr // 2: n - 9 * sr] = 0
    y = np.tanh(6 * x)
    b, a = butter(1, 3000 / (sr / 2))
    y = (lfilter(b, a, y) * 0.5).astype(np.float32)
    directory.mkdir(parents=True, exist_ok=True)
    # 24-bit PCM like the files users record/export (float WAVs are only read by the newest trainers)
    sf.write(str(directory / "in.wav"), x, sr, subtype="PCM_24")
    sf.write(str(directory / "out.wav"), y, sr, subtype="PCM_24")


def run_case(name: str, job: dict, args, wav_dir: Path, expect_two_stage: bool) -> dict:
    folder = Path(args.workdir) / f"caso-{re.sub(r'[^a-z0-9]+', '-', name.lower())}"
    (folder / "_lavoro").mkdir(parents=True, exist_ok=True)
    (folder / "_lavoro" / "job.json").write_text(json.dumps(job), encoding="utf-8")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
           "PYTHONPATH": os.pathsep.join(p for p in (str(HOOK_DIR), os.environ.get("PYTHONPATH", "")) if p),
           # short scratch path for checkpoints, like the server gives production workers (see worker.py)
           "NAM_TRAIN_DIR": str(Path(tempfile.gettempdir()) / f"nt-{os.getpid()}-{folder.name[-12:]}")}
    log_path = folder / "worker.log"
    started = time.time()
    say("sel.starting_worker", name=name)
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "worker.py"), str(folder), str(wav_dir), "--trainer-path", str(Path(args.path).resolve())],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        try:
            code = proc.wait(timeout=CASE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = -1
    text = log_path.read_text(encoding="utf-8", errors="replace")
    tail = "\n".join(l.rstrip("\r") for l in text.splitlines()[-12:])

    def fail(reason: dict) -> dict:
        return {"name": name, "ok": False, "error": M("sel.case_error", reason=reason, tail=tail)}

    if code != 0:
        return fail(M("sel.worker_timeout" if code == -1 else "sel.worker_exit", code=code))
    match = re.search(r"^Trainer: (\S+) \| nam (\S+) from (.+)$", text, re.MULTILINE)
    if not match:
        return fail(M("sel.no_trainer_line"))
    if Path(match.group(3).strip()).resolve() != (Path(args.path).resolve() / "nam"):
        return fail(M("sel.wrong_package_worker", path=match.group(3)))
    model_file = folder / f"{job['basename']}.nam"
    if not model_file.exists():
        return fail(M("sel.no_model"))
    try:
        model = json.loads(model_file.read_text(encoding="utf-8"))
        assert model["architecture"] and model["version"] and "weights" in model and "config" in model
    except Exception as e:  # noqa: BLE001
        return fail(M("sel.invalid_model", detail=f"{type(e).__name__}: {e}"))
    try:
        progress = json.loads((folder / "_lavoro" / "progress.json").read_text(encoding="utf-8"))
        if expect_two_stage and not (progress.get("stages") == 2 and progress.get("stage") == 2):
            return fail(M("sel.not_two_stage", progress=str(progress)))
    except (OSError, ValueError):
        return fail(M("sel.no_progress"))
    say("sel.case_ok", name=name, seconds=round(time.time() - started), architecture=model["architecture"])
    return {"name": name, "ok": True, "seconds": round(time.time() - started), "architecture": model["architecture"]}


def pick_arch(catalog: dict, group: str, prefer: str):
    entries = [e for e in catalog["entries"] if e["group"] == group]
    for e in entries:
        if e["value"] == prefer:
            return e
    return entries[0] if entries else None


def main_run(args) -> int:
    pkg = Path(args.path).resolve()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def finish(ok: bool, error=None, cases=None, catalog=None) -> int:
        emit_result({"ok": ok, "error": error, "cases": cases or [], "has_catalog": catalog is not None})
        return 0 if ok else 1

    if not (pkg / "nam" / "__init__.py").exists():
        return finish(False, M("sel.no_init", pkg=str(pkg)))

    say("sel.checking")
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "introspect", "--trainer", args.trainer,
                            "--path", str(pkg)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONUTF8": "1"}, timeout=600)
    line = next((l for l in reversed(child.stdout.splitlines()) if l.startswith("RESULT ")), None)
    if child.returncode != 0 or line is None:
        return finish(False, M("sel.import_failed", detail="\n".join((child.stderr or child.stdout).splitlines()[-12:])))
    info = json.loads(line[len("RESULT "):])
    for l in child.stdout.splitlines():
        if l.startswith("[selftest]"):
            print(l, flush=True)
    if info["problems"]:
        return finish(False, M("sel.problems", items=info["problems"]))
    if Path(info["file"]).parent != pkg / "nam":
        return finish(False, M("sel.other_package", path=info["file"]))
    say("sel.imported", version=info["version"])
    if info["version"] != args.expect_version:
        # Only a note: the release identity is guaranteed by the tag / sha256 it was downloaded with, and some
        # projects forget to bump the version inside the package (Reloaded v1.0.0 reports 0.1.dev335).
        say("sel.declared_version", declared=info["version"], expected=args.expect_version)

    catalog = info["catalog"]
    wav_dir = workdir / "wav"
    make_wavs(wav_dir, seconds=20 if hardware.variant() == "cpu" else 30)
    base = {"input_rel": "in.wav", "output_rel": "out.wav", "latency": 0, "threshold_esr": None, "ignore_checks": True,
            "epochs": 1, "batch_size": 16, "metadata": {"name": "selftest"}}
    cases = []
    if args.trainer == "official":
        cases.append(("official", {**base, "backend": "official", "architecture": None, "basename": "selftest"}, False))
    else:
        a2 = pick_arch(catalog, "a2", "A2 Full+Lite")
        light = pick_arch(catalog, "light", "standard") or pick_arch(catalog, "extended", "complex")
        if a2:
            cases.append((f"{a2['value']} (2 stages)", {
                **base, "backend": "reloaded", "architecture": a2["value"], "basename": "selftest", "batch_size": 32,
                "lr": a2["lr"], "lr_decay": a2["lr_decay"], "lr_scheduler": "exponential", "fit_mrstft": True,
                "stage_mode": "two_stage", "stage2_epochs": 1, "stage2_lr": 0.0025, "stage2_lr_decay": 0.6625,
                "stage2_lr_scheduler": "reduce_on_plateau", "stage2_focus": "Lows"}, True))
        if light:
            cases.append((f"{light['value']}", {
                **base, "backend": "reloaded", "architecture": light["value"], "basename": "selftest", "batch_size": 32,
                "lr": light["lr"], "lr_decay": light["lr_decay"], "lr_scheduler": "exponential", "fit_mrstft": True,
                "stage_mode": "single_stage"}, False))
        if not cases:
            return finish(False, M("sel.no_cases"))

    results = []
    for name, job, two_stage in cases:
        result = run_case(name, job, args, wav_dir, two_stage)
        results.append(result)
        if not result["ok"]:
            return finish(False, M("sel.case_failed", name=name, reason=result["error"]), results)
    if args.emit_catalog and catalog is not None:
        Path(args.emit_catalog).write_text(json.dumps({**catalog, "version": info["version"],
                                                       "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1),
                                           encoding="utf-8")
        say("sel.catalog_written", count=len(catalog["entries"]))
    return finish(True, None, results, catalog)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("run", "introspect"):
        p = sub.add_parser(mode)
        p.add_argument("--trainer", choices=("official", "reloaded"), required=True)
        p.add_argument("--path", required=True)
        if mode == "run":
            p.add_argument("--workdir", required=True)
            p.add_argument("--expect-version", required=True)
            p.add_argument("--emit-catalog")
    args = parser.parse_args()
    return main_run(args) if args.mode == "run" else introspect(args)


if __name__ == "__main__":
    sys.exit(main())
