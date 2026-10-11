"""Final step of the NAM Suite installer for the trainer (run by the installer with the suite's own Python, standard library only).

The installer has already copied the files: <trainer>\\site holds the packages of the chosen PyTorch build. This script
  1. checks that everything imports and, for NVIDIA / AMD, that PyTorch can really use the GPU (a small matrix product);
  2. if AMD was chosen and the GPU is not usable, swaps the ROCm build for the CPU build that came along (<trainer>\\_cpu_fallback);
  3. writes nam-web\\hardware.json (which accelerator this installation uses, the language chosen in the installer);
  4. removes the leftovers of the installation (_cpu_fallback).

Exit code: 0 ok | 10 ok but switched to the CPU build (AMD GPU not usable) | 11 ok but PyTorch does not see the GPU (old driver?) | 20 a package does not import.
The log is written to <trainer>\\nam-web\\installazione.log.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHECK = r"""
import sys, torch, torchaudio, pytorch_lightning, fastapi, uvicorn, soundfile, scipy, matplotlib, librosa, multipart, packaging
state, gpu = 'nocuda', '-'
if torch.cuda.is_available():
    try:
        gpu = torch.cuda.get_device_name(0)
        x = torch.randn(256, 256, device='cuda')
        float((x @ x).sum().item())
        torch.cuda.synchronize()
        state = 'cuda'
    except Exception as e:
        state, gpu = 'cudaerr', type(e).__name__ + ': ' + str(e)[:120].replace('|', '/').replace('\n', ' ')
print('IMPORT_OK', torch.__version__, state, gpu, sep='|')
"""
ROCM_PACKAGES = ("torch", "torchaudio", "rocm", "rocm_sdk_core", "rocm_sdk_devel", "rocm_sdk_libraries_custom")

log_lines = []


def say(text):
    log_lines.append(text)
    print(text, flush=True)


def run_check(env):
    proc = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, env=env, timeout=900,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    line = next((l for l in proc.stdout.splitlines() if l.startswith("IMPORT_OK|")), None)
    if not line:
        say("check failed: " + "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:]))
        return None
    return line.split("|")


def uninstall(site, names):
    """Remove installed packages using their RECORD files (no pip needed)."""
    for dist in list(site.glob("*.dist-info")):
        base = dist.name[:-len(".dist-info")].rsplit("-", 1)[0].lower().replace("-", "_")
        if base not in names:
            continue
        record = dist / "RECORD"
        if record.exists():
            for row in record.read_text(encoding="utf-8", errors="replace").splitlines():
                rel = row.rsplit(",", 2)[0].strip('"')
                target = site / rel
                if target.is_file():
                    try:
                        target.unlink()
                    except OSError:
                        pass
        shutil.rmtree(dist, ignore_errors=True)
    for name in names:          # empty package folders the RECORD did not list
        shutil.rmtree(site / name, ignore_errors=True)
    for entry in list(site.iterdir()):      # the ROCm packages keep their files in folders named _rocm_sdk_*, rocm_sdk ...
        if entry.is_dir() and entry.name.lower().lstrip("_").startswith("rocm"):
            shutil.rmtree(entry, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--variant", choices=("nvidia", "amd", "cpu"), required=True)
    ap.add_argument("--gpu", default="")
    ap.add_argument("--cpu", default="")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    trainer = Path(args.trainer)
    site = trainer / "site"
    fallback = trainer / "_cpu_fallback"
    env = {**os.environ, "PYTHONPATH": str(site), "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONHOME", None)
    variant, requested, code, gpu_name = args.variant, args.variant, 0, args.gpu
    say(f"NAM Suite setup {time.strftime('%Y-%m-%d %H:%M:%S')} - variant {variant}, language {args.lang}")

    parts = run_check(env)
    if variant == "amd" and (not parts or parts[2] != "cuda"):
        say("AMD: " + (f"the GPU is not usable ({parts[2]}: {parts[3]})" if parts else "PyTorch ROCm does not start") + " - switching to the CPU build")
        uninstall(site, ROCM_PACKAGES)
        if not fallback.is_dir():
            say("the CPU fallback is missing")
            return 20
        shutil.copytree(fallback, site, dirs_exist_ok=True)
        variant, code, gpu_name = "cpu", 10, ""
        parts = run_check(env)
    if not parts:
        return 20
    say(f"PyTorch {parts[1]}: ok, {parts[2]} {parts[3]}")
    if variant != "cpu" and parts[2] != "cuda" and code == 0:
        code = 11
    if parts[2] == "cuda":
        gpu_name = parts[3]

    hardware = {"variant": variant, "gpu_name": gpu_name if variant != "cpu" else "", "cpu_name": args.cpu, "requested": requested,
                "language": args.lang if args.lang in ("it", "en") else "en", "installed": time.strftime("%Y-%m-%d %H:%M:%S")}
    web = trainer / "nam-web"
    web.mkdir(parents=True, exist_ok=True)
    (web / "hardware.json").write_text(json.dumps(hardware, indent=1), encoding="utf-8")
    shutil.rmtree(fallback, ignore_errors=True)
    return code


if __name__ == "__main__":
    status = 20
    try:
        status = main()
    except Exception as exc:                      # the installer must get an exit code, not a traceback window
        say(f"error: {type(exc).__name__}: {exc}")
    try:
        log = Path(sys.argv[sys.argv.index("--trainer") + 1]) / "nam-web" / "installazione.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    except Exception:
        pass
    sys.exit(status)
