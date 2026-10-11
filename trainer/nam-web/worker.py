"""Runs one NAM training job in its own process.

Usage: python worker.py <model_folder> <wav_root> [--trainer-path <dir>]

--trainer-path replaces ../trainers/<backend> (used by selftest.py to try a freshly
downloaded release before it is installed).

<model_folder> is the folder of the model inside "NAM Generati". It reads
<model_folder>/_lavoro/job.json (WAVs are referenced relative to <wav_root>) and writes:
  <name>.nam, <name>.png      exported model and validation plot, in <model_folder>
  _lavoro/progress.json       live progress (epoch, ESR, ETA)
  _lavoro/result.json         final outcome
  _lavoro/train/              checkpoints and logs while training; removed at the end
Creating _lavoro/stop asks training to stop early and still export the best model.

job.json "backend" picks the trainer: the worker puts ../trainers/<backend> (a folder holding
an unmodified `nam` package, replaced by updater.py on new releases) first on sys.path so that
it shadows any other `nam`. That is why each job runs in a fresh process. If
../trainers/official is missing, the `nam` installed by pip in the venv is used instead.
"""

import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import fsutil
import hardware
import paths
import priority

os.environ.setdefault("MPLBACKEND", "Agg")

WORK = "_lavoro"
TRAINERS_DIR = paths.TRAINERS_DIR


def write_json(path: Path, data: dict) -> None:
    fsutil.write_text_atomic(path, json.dumps(data))


def exit_when_server_dies(server_pid: int) -> None:
    """Don't leave a training run hogging the GPU if the server is killed."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, server_pid)  # SYNCHRONIZE
        if handle:
            kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
    else:
        while os.getppid() == server_pid:
            time.sleep(2)
    os._exit(1)


def train_kwargs(job: dict, user_metadata) -> dict:
    kwargs = dict(
        epochs=int(job["epochs"]),
        latency=job.get("latency"),
        batch_size=int(job["batch_size"]),
        silent=True,
        save_plot=True,
        modelname="model",
        ignore_checks=bool(job.get("ignore_checks")),
        threshold_esr=job.get("threshold_esr"),
        user_metadata=user_metadata,
    )
    if job["backend"] == "reloaded":
        extra = dict(
            architecture=job["architecture"],
            lr=job.get("lr"),
            lr_decay=job.get("lr_decay"),
            lr_scheduler_type=job.get("lr_scheduler"),
            fit_mrstft=job.get("fit_mrstft"),
            stage_mode=job.get("stage_mode"),
            stage2_epochs=job.get("stage2_epochs"),
            stage2_lr=job.get("stage2_lr"),
            stage2_lr_decay=job.get("stage2_lr_decay"),
            stage2_lr_scheduler_type=job.get("stage2_lr_scheduler"),
            stage2_focus=job.get("stage2_focus"),
            checkpoint_save_mode="Minimal",
            seed=0,
        )
        kwargs.update({k: v for k, v in extra.items() if v is not None})  # None = the trainer's default
    return kwargs


def main(folder: Path, wav_root: Path, trainer_path: Path = None) -> int:
    work = folder / WORK
    job = json.loads((work / "job.json").read_text(encoding="utf-8"))
    job.setdefault("backend", "official")
    basename = job["basename"]
    epochs = int(job["epochs"])
    two_stage = job["backend"] == "reloaded" and job.get("stage_mode") == "two_stage" and int(job.get("stage2_epochs") or 0) > 0
    total_epochs = epochs + (int(job["stage2_epochs"]) if two_stage else 0)

    trainer_dir = Path(trainer_path) if trainer_path else TRAINERS_DIR / job["backend"]
    trainer_dir = trainer_dir.resolve()
    use_folder = (trainer_dir / "nam" / "__init__.py").exists()
    if use_folder:
        sys.path.insert(0, str(trainer_dir))
    elif job["backend"] != "official":  # only the official trainer has a pip-installed fallback
        raise RuntimeError(f"Trainer '{job['backend']}' is not installed: {trainer_dir} is missing")

    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # an operation without a Metal kernel runs on the CPU instead of failing
    import pytorch_lightning as pl
    import torch
    import nam
    from nam.models.metadata import UserMetadata
    from nam.train import core

    try:
        print(priority.boost_process())       # top priority, no power throttling, the fast cores: see priority.py
    except Exception as e:                    # best effort: a training is never stopped for it
        print(f"priority boost: failed ({type(e).__name__}: {e})")
    nam_file = Path(nam.__file__).resolve()
    if use_folder and trainer_dir not in nam_file.parents:
        raise RuntimeError(f"The wrong nam package was loaded: {nam_file}")
    print(f"Trainer: {job['backend']} | nam {nam.__version__} from {nam_file.parent}")

    # Device: an accelerator (CUDA on NVIDIA, HIP/ROCm on AMD: both answer to torch.cuda; Metal on Apple Silicon) when PyTorch sees one.
    # Training on the CPU is only done when this installation was set up for it (hardware.json variant "cpu");
    # for a GPU install a missing GPU is a problem to fix (driver, restart), not something to train slowly through.
    hw = hardware.load()
    if torch.cuda.is_available():
        hip = getattr(torch.version, "hip", None)
        print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}{f' | HIP {hip}' if hip else ''}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print(f"GPU: Apple Metal (MPS) | torch {torch.__version__}")   # the trainers' own device selection picks it (core.py: accelerator "mps")
    elif hw["variant"] == "cpu":
        print(f"CPU: {hardware.cpu_name()} | torch {torch.__version__} | no GPU: training will be slow")
    else:
        print(f"ERROR: this installation is set up for a {hw['variant']} GPU but PyTorch does not see one.")
        write_json(work / "result.json", {"ok": False, "error": {"k": "job.no_gpu", "p": {"variant": hw["variant"]}}})
        return 2

    class Progress(pl.Callback):
        def __init__(self):
            self.t0 = time.time()
            self.last_flush = 0.0
            self.offset = 0  # epochs of the stages that already finished
            self.state = {"epoch": 0, "epochs": total_epochs, "stage": 0, "stages": 2 if two_stage else 1,
                          "batch": 0, "batches": 0, "esr": None, "best_esr": None}

        def flush(self, force=False):
            now = time.time()
            if not force and now - self.last_flush < 1.0:
                return
            self.last_flush = now
            s = self.state
            done = s["epoch"] + (s["batch"] / s["batches"] if s["batches"] else 0)
            elapsed = now - self.t0
            s["elapsed_s"] = round(elapsed)
            s["eta_s"] = round(elapsed / done * (total_epochs - done)) if done > 0.05 else None
            try:
                write_json(work / "progress.json", s)
            except OSError:
                pass  # the file stayed locked by a reader for seconds: skip this update, the next one will do; never stop a training for it

        def on_fit_start(self, trainer, pl_module):
            self.state["stage"] = min(self.state["stage"] + 1, self.state["stages"])

        def on_fit_end(self, trainer, pl_module):
            self.offset += trainer.max_epochs or 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.state.update(epoch=self.offset + trainer.current_epoch, batch=batch_idx + 1,
                              batches=trainer.num_training_batches)
            if (work / "stop").exists():
                trainer.should_stop = True
            self.flush()

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            esr = trainer.callback_metrics.get("ESR")
            if esr is not None:
                esr = float(esr)
                self.state["esr"] = esr
                best = self.state["best_esr"]
                self.state["best_esr"] = esr if best is None else min(best, esr)
            self.state.update(epoch=self.offset + trainer.current_epoch + 1, batch=0)
            self.flush(force=True)

    progress = Progress()
    original_get_callbacks = core.get_callbacks

    def get_callbacks(*args, **kwargs):
        result = original_get_callbacks(*args, **kwargs)
        if isinstance(result, tuple):  # Reloaded: (callbacks, preferred_checkpoint_callback, epoch_time_logger)
            callbacks, *rest = result
            return (list(callbacks) + [progress], *rest)
        return result + [progress]  # official: list of callbacks

    core.get_callbacks = get_callbacks

    metadata = {k: v for k, v in job.get("metadata", {}).items() if v not in (None, "")}
    user_metadata = UserMetadata(**metadata)

    # Scratch folder for checkpoints and logs. The Reloaded trainer nests them deeply
    # (train/model/stage_1/<architecture>/<epochs>/<hyper-parameters>/checkpoints/...), which overflows
    # Windows' 260 character path limit inside a long model folder name; NAM_TRAIN_DIR (set by the
    # server, a short path under nam-web/tmp) avoids that. Always removed when done.
    train_dir = Path(os.environ["NAM_TRAIN_DIR"]) if os.environ.get("NAM_TRAIN_DIR") else work / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    input_path = str(wav_root / job["input_rel"])
    output_path = str(wav_root / job["output_rel"])
    try:
        out = core.train(input_path, output_path, str(train_dir), **train_kwargs(job, user_metadata))

        if out is None or out.model is None or getattr(out, "aborted", False):
            print("No model exported: the data checks failed or training was interrupted (see above).")
            write_json(work / "result.json", {"ok": False, "error": {"k": "job.checks_failed"}})
            return 3

        print("Exporting the model...")
        out.model.net.export(
            folder,
            basename=basename,
            user_metadata=user_metadata,
            other_metadata={"training": out.metadata.model_dump()},
        )
        plots = sorted(train_dir.glob("*.png"))
        if plots:
            shutil.move(plots[0], folder / f"{basename}.png")
    finally:
        shutil.rmtree(train_dir, ignore_errors=True)  # checkpoints and tensorboard logs

    esr = out.metadata.validation_esr
    write_json(work / "result.json", {"ok": True, "esr": esr, "stopped_early": (work / "stop").exists()})
    print(f"Done. Validation ESR = {esr}")
    return 0


if __name__ == "__main__":
    model_folder, wav_root = Path(sys.argv[1]), Path(sys.argv[2])
    override = Path(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[3] == "--trainer-path" else None
    if os.environ.get("NAM_WEB_SERVER_PID"):
        threading.Thread(target=exit_when_server_dies, args=(int(os.environ["NAM_WEB_SERVER_PID"]),), daemon=True).start()
    try:
        code = main(model_folder, wav_root, override)
    except BaseException as e:  # noqa: BLE001 - report everything to the server
        traceback.print_exc()
        write_json(model_folder / WORK / "result.json",
                   {"ok": False, "error": {"k": "job.error_raw", "p": {"detail": f"{type(e).__name__}: {e}"}}})
        code = 1
    sys.stdout.flush()
    sys.exit(code)
