r"""Builds the NAM Suite installer (Inno Setup): trainer + capture in one classic Windows setup.

    python build.py all            # tools + runtime + trainer + variants + capture + extras + compile
    python build.py <step>         # tools | launcher | runtime | trainer | variants | capture | extras | iss
    python build.py iss --variants cpu      # quick test build: only the CPU PyTorch build goes in (small, fast)

Sources stay where they are (nothing is copied back):
    trainer   ..\trainer   (nam-web, trainers)      pinned packages: suite\locks      wheel cache: suite\cache
    capture   ..\capture   (capture_tool, packaging\windows\NamCaptureApp exe, built with MSBuild)
What is staged (cache\stage) and what the installer does with it: see installer\NAMSuite.iss.

    runtime\            portable Python 3.12 (shared by both apps; no packages of its own)
    trainer\            nam-web, trainers, site\ (Python packages, pre-installed: no pip at install time), NAM Trainer.exe
    capture\            NAM Capture.exe (+ WebView2Loader.dll), capture_tool, site\ (its own, lean packages)
    variants\<v>\       the PyTorch build of each kind of PC (nvidia | amd | cpu) - the installer copies the one that fits into trainer\site
    extras\             Visual C++ runtime, WebView2 runtime (only installed when missing), helper programs
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                             # the nam-suite repository
TRAINER = ROOT / "trainer"
CAP = ROOT / "capture"
LOCKS = HERE / "locks"                         # pinned package versions (common + one PyTorch build per kind of PC)
VERSION = (HERE / "VERSION").read_text(encoding="utf-8").strip()
CACHE = HERE / "cache"
STAGE = CACHE / "stage"
OUTPUT = Path(os.environ.get("NAM_SUITE_OUT") or HERE / "output")
CSC = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
ISCC = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe"
VARIANTS = ("nvidia", "amd", "cpu")
PIN = json.loads((HERE / "python-pin.json").read_text(encoding="utf-8"))
TARGET = "windows" if sys.platform == "win32" else "macos"      # what is being built (the host decides today: Windows installer on Windows, macOS package on a Mac)


def standalone_python(target: str = None) -> Path:
    """The pinned python-build-standalone for `target` (downloaded once into cache\python, checked against the pinned sha256).
    Returns the folder that holds python.exe (Windows) or bin/python3 (macOS). It is both the Python that ships and the one that runs pip here."""
    target = target or TARGET
    info = PIN[target]
    root = CACHE / "python" / f"{PIN['version']}-{PIN['tag']}-{target}"
    inner = root / "python"
    if (inner / ("python.exe" if target == "windows" else "bin/python3")).exists():
        return inner
    archive = CACHE / "python" / info["file"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        url = "https://github.com/astral-sh/python-build-standalone/releases/download/{}/{}".format(PIN["tag"], info["file"].replace("+", "%2B"))
        say(f"downloading {info['file']}")
        urllib.request.urlretrieve(url, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != info["sha256"]:
        archive.unlink()
        raise SystemExit(f"{info['file']}: sha256 {digest} does not match the pin in python-pin.json")
    rmtree(root)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    return inner


def pip_python() -> Path:
    """Python that runs pip / compileall while building ($NAM_PIP_PY overrides; same minor version as the shipped Python is what matters)."""
    override = os.environ.get("NAM_PIP_PY")
    return Path(override) if override else standalone_python() / ("python.exe" if TARGET == "windows" else "bin/python3")


CAPTURE_EXE = CAP / "packaging" / "windows" / "NamCaptureApp" / "x64" / "Release" / "NamCaptureApp.exe"
CAPTURE_LOCK = HERE / "installer" / "requirements-capture.lock"
WEBVIEW2_URL = "https://go.microsoft.com/fwlink/?linkid=2124701"     # Evergreen standalone installer, x64
WEBVIEW2_EXE = CACHE / "MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
REDIST = CACHE / "vc_redist.x64.exe"


def say(text):
    print(f"[suite] {text}", flush=True)


def run(cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def rmtree(path):
    if path.exists():
        shutil.rmtree(path)


def du(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# ------------------------------------------------------------------------------------------ helper programs
def step_tools():
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "extras").mkdir(exist_ok=True)
    (STAGE / "trainer").mkdir(exist_ok=True)
    launcher = HERE / "launcher"
    manifest = f"/win32manifest:{launcher / 'app.manifest'}"
    common = ["/nologo", "/platform:x64", "/optimize+", "/reference:System.dll", "/reference:System.Core.dll"]

    def csc(out, target, sources, icon=None, refs=()):
        cmd = [CSC, *common, f"/target:{target}", f"/out:{out}", manifest, *[f"/reference:{r}" for r in refs], *sources]
        if icon:
            cmd.insert(1, f"/win32icon:{icon}")
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
        if proc.returncode:
            raise SystemExit(f"csc failed for {out.name}:\n{proc.stdout}{proc.stderr}")
        say(f"{out.name}: {out.stat().st_size / 1024:.0f} KB")

    csc(STAGE / "extras" / "gpu-detect.exe", "exe", [launcher / "GpuDetect.cs"], None, ["System.Management.dll"])
    shutil.copy2(HERE / "installer" / "suite_setup.py", STAGE / "extras" / "suite_setup.py")


# ------------------------------------------------------------------------------------------ runtime
def step_runtime():
    dst = STAGE / "runtime"
    rmtree(dst)
    base = standalone_python("windows")
    drop_top = {"Scripts", "include", "libs", "share"}                 # launchers of pip & co. point at this PC; headers/import libraries are for compiling
    drop_lib = {"test", "idlelib", "turtledemo", "__pycache__", "ensurepip", "lib2to3"}

    def ignore(directory, names):
        d = Path(directory).resolve()
        skip = {n for n in names if n == "__pycache__" or n.endswith(".pyc")}
        if d == base.resolve():
            skip |= {n for n in names if n in drop_top}
        if d == (base / "Lib").resolve():
            skip |= {n for n in names if n in drop_lib or n == "site-packages"}
        return skip

    shutil.copytree(base, dst, ignore=ignore)
    (dst / "Lib" / "site-packages").mkdir(exist_ok=True)
    say(f"runtime: {du(dst) / 1e6:.0f} MB ({subprocess.run([str(dst / 'python.exe'), '--version'], capture_output=True, text=True).stdout.strip()})")


# ------------------------------------------------------------------------------------------ python packages
def pip_target(lock, wheelhouses, target):
    rmtree(target)
    cmd = [pip_python(), "-m", "pip", "install", "--no-index", "--no-deps", "--disable-pip-version-check", "--no-compile", "--no-warn-script-location",
           "--target", target, "-r", lock]
    for w in wheelhouses:
        cmd[6:6] = ["--find-links", w]
    run(cmd)
    rmtree(target / "bin")                      # console-script launchers: not used, and they point at this PC
    for tests in [d for d in target.rglob("*") if d.is_dir() and d.name in ("tests", "test") and d.parent != target]:
        shutil.rmtree(tests, ignore_errors=True)   # the packages' own test suites: never imported, and their long names break Windows' 260-char path limit


def precompile(target):
    """pyc files next to the sources, validated by hash (file times change when the installer copies them)."""
    run([pip_python(), "-m", "compileall", "-q", "-j", "0", "--invalidation-mode", "unchecked-hash", target])


def step_trainer():
    t = STAGE / "trainer"
    for name in ("nam-web", "trainers", "site"):
        rmtree(t / name)
    rmtree(STAGE / "trainers-bundled")
    shutil.copytree(TRAINER / "nam-web", t / "nam-web", ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "tmp", "cache", "data", "*.log", "settings.json", "hardware.json", "start.ps1", "stop.ps1", "calibration.json"))
    # the two NAM trainers travel apart (the installer keeps the ones already installed: the page can update them)
    shutil.copytree(TRAINER / "trainers", STAGE / "trainers-bundled", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for doc in ("LICENSE", "THIRD_PARTY.md"):
        if (ROOT / doc).exists():
            shutil.copy2(ROOT / doc, t / doc)
    say("trainer: pre-installing the common packages (no network)...")
    pip_target(LOCKS / "requirements.lock", [CACHE / "wheelhouse"], t / "site")
    precompile(t / "site")
    leaks = personal_paths(t) + personal_paths(STAGE / "trainers-bundled")
    if leaks:
        raise SystemExit("the package mentions this PC's user folder:\n  " + "\n  ".join(leaks[:15]))
    say(f"trainer: {du(t) / 1e6:.0f} MB")


def personal_paths(root):
    home = str(Path.home())
    needles = [home.encode(), home.replace("\\", "/").encode()]
    hits = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() in (".exe", ".dll", ".pyd", ".pyc", ".wav", ".png", ".ico", ".zip", ".so", ".lib", ".whl", ".ttf", ".nam"):
            continue
        if "site" in p.relative_to(root).parts[:1] and p.suffix.lower() not in (".py", ".txt", ".json", ".cfg", ".ini", ".md"):
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:4096]:
            continue
        for n in needles:
            if n in data:
                hits.append(f"{p.relative_to(root)} -> {n.decode()}")
                break
    return hits


def step_variants(only=None):
    base = STAGE / "variants"
    for v in only or VARIANTS:
        target = base / v
        say(f"PyTorch {v}: installing...")
        pip_target(LOCKS / f"requirements-{v}.lock", [CACHE / f"wheelhouse-{v}", CACHE / "wheelhouse"], target)
        precompile(target)
        say(f"PyTorch {v}: {du(target) / 1e9:.2f} GB")


# ------------------------------------------------------------------------------------------ capture
def step_capture():
    c = STAGE / "capture"
    rmtree(c)
    c.mkdir(parents=True)
    if not CAPTURE_EXE.exists():
        raise SystemExit(f"{CAPTURE_EXE} is missing: build NamCaptureApp first (packaging\\windows\\NamCaptureApp, MSBuild Release x64)")
    shutil.copy2(CAPTURE_EXE, c / "NAM Capture.exe")
    shutil.copy2(CAPTURE_EXE.with_name("WebView2Loader.dll"), c / "WebView2Loader.dll")
    shutil.copytree(CAP / "capture_tool", c / "capture_tool", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tmp", "settings.json", "calibration.json"))
    for doc in ("LICENSE", "THIRD_PARTY.md"):                  # (the manual is not part of the installer)
        if (ROOT / doc).exists():
            shutil.copy2(ROOT / doc, c / doc)
    wheelhouse = CACHE / "wheelhouse-capture"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    say("capture: fetching the wheels it needs (only those not already cached)...")
    run([pip_python(), "-m", "pip", "download", "--disable-pip-version-check", "--no-deps", "--only-binary=:all:", "-r", CAPTURE_LOCK,
         "-d", wheelhouse, "--find-links", CACHE / "wheelhouse"])
    pip_target(CAPTURE_LOCK, [wheelhouse, CACHE / "wheelhouse"], c / "site")
    precompile(c / "site")
    leaks = personal_paths(c)
    if leaks:
        raise SystemExit("the capture package mentions this PC's user folder:\n  " + "\n  ".join(leaks[:15]))
    say(f"capture: {du(c) / 1e6:.0f} MB")


# ------------------------------------------------------------------------------------------ extras
def signed_by_microsoft(path):
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"$s = Get-AuthenticodeSignature -LiteralPath '{path}'; \"$($s.Status)|$($s.SignerCertificate.Subject)\""],
                         capture_output=True, text=True).stdout.strip()
    say(f"signature of {path.name}: {out}")
    return out.startswith("Valid|") and "Microsoft Corporation" in out


def step_extras():
    extras = STAGE / "extras"
    extras.mkdir(parents=True, exist_ok=True)
    if not REDIST.exists() or not signed_by_microsoft(REDIST):
        raise SystemExit(f"{REDIST} is missing or not signed by Microsoft (build the 1.0 installer cache first)")
    shutil.copy2(REDIST, extras / "vc_redist.x64.exe")
    if not WEBVIEW2_EXE.exists():
        say(f"downloading the WebView2 runtime installer ({WEBVIEW2_URL})...")
        run(["curl.exe", "-sL", "-o", WEBVIEW2_EXE, WEBVIEW2_URL])
    if not signed_by_microsoft(WEBVIEW2_EXE):
        WEBVIEW2_EXE.unlink(missing_ok=True)
        raise SystemExit("the WebView2 installer has no valid Microsoft signature: removed")
    shutil.copy2(WEBVIEW2_EXE, extras / WEBVIEW2_EXE.name)
    for name in ("nam-suite.ico", "nam-trainer.ico", "nam-capture.ico"):
        shutil.copy2(HERE / "assets" / name, extras / name)
    for guide in ("LEGGIMI.txt", "README-EN.txt"):
        shutil.copy2(HERE / "docs" / guide, extras / guide)
    say(f"extras: {du(extras) / 1e6:.0f} MB")


# ------------------------------------------------------------------------------------------ compile
def longest_installed_path():
    """Longest path (relative to the install folder) the installer will create: variants\<v> land in trainer\site (the CPU build of an
    AMD install also in trainer\_cpu_fallback). Windows allows 259 characters in all: the wizard refuses install folders longer than the rest."""
    best = (0, None)
    for p in STAGE.rglob("*"):
        if not p.is_file():
            continue
        parts = p.relative_to(STAGE).parts
        if parts[0] == "extras":
            continue
        if parts[0] == "variants":
            tails = [("trainer", "site", *parts[2:])]
            if parts[1] == "cpu":
                tails.append(("trainer", "_cpu_fallback", *parts[2:]))
        elif parts[0] == "trainers-bundled":
            tails = [("trainer", "trainers", *parts[1:])]
        else:
            tails = [parts]
        for t in tails:
            n = len("\\".join(t))
            if n > best[0]:
                best = (n, "\\".join(t))
    return best


def step_iss(only=None, test_data=None):
    if not ISCC.exists():
        raise SystemExit(f"Inno Setup not found: {ISCC}")
    OUTPUT.mkdir(exist_ok=True)
    for old in OUTPUT.glob("NAM-Suite-Setup*"):
        old.unlink()
    longest, longest_path = longest_installed_path()
    max_dir = 259 - longest - 2
    say(f"longest installed path: {longest} chars -> install folders up to {max_dir} chars ({longest_path})")
    if max_dir < 60:
        raise SystemExit("a file path in the package is too long: prune it in build.py")
    variants = [v for v in VARIANTS if (STAGE / "variants" / v).is_dir() and (only is None or v in only)]
    if not variants:
        raise SystemExit("no variant staged: python build.py variants")
    extra_trainer = max((du(STAGE / "variants" / v) for v in variants), default=0)      # the wizard adds the biggest PyTorch build to the trainer's size
    defines = [f"/DExtraTrainer={extra_trainer}", f"/DMaxDirLen={max_dir}", f"/DVersion={VERSION}", f"/DStage={STAGE}", f"/DAssets={HERE / 'assets'}", f"/DOut={OUTPUT}",
               "/DHasNvidia=" + ("1" if "nvidia" in variants else "0"), "/DHasAmd=" + ("1" if "amd" in variants else "0"),
               "/DHasCpu=" + ("1" if "cpu" in variants else "0")]
    if test_data:          # tests only: the uninstaller cleans this folder instead of the real %LOCALAPPDATA%\NAM Capture
        defines.append(f"/DCaptureData={test_data}")
    started = time.time()
    subprocess.run([str(ISCC), "/Q", *defines, str(HERE / "installer" / "NAMSuite.iss")], check=True)
    files = sorted(OUTPUT.glob("NAM-Suite-Setup*"))
    for f in files:
        say(f"{f.name}: {f.stat().st_size / 1e6:.0f} MB")
    (OUTPUT / "SHA256SUMS.txt").write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest() if f.stat().st_size < 400e6 else sha_big(f)} *{f.name}\n" for f in files), encoding="utf-8")
    say(f"installer built in {time.time() - started:.0f}s")


def sha_big(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def dotnet() -> Path:
    """The .NET 8 SDK that builds the launcher, installed by Microsoft's own script into cache\dotnet (nothing is installed on the PC)."""
    home = CACHE / ("dotnet" if TARGET == "windows" else "dotnet-macos")     # a cache folder carried over from the other system must not be mistaken for this one
    exe = home / ("dotnet.exe" if TARGET == "windows" else "dotnet")
    if exe.exists():
        return exe
    scripts = CACHE / "dotnet-install"
    scripts.mkdir(parents=True, exist_ok=True)
    if TARGET == "windows":
        script = scripts / "dotnet-install.ps1"
        urllib.request.urlretrieve("https://dot.net/v1/dotnet-install.ps1", script)
        if not signed_by_microsoft(script):
            script.unlink()
            raise SystemExit("dotnet-install.ps1 has no valid Microsoft signature")
        run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-Channel", "8.0", "-InstallDir", CACHE / "dotnet", "-NoPath"])
    else:
        script = scripts / "dotnet-install.sh"
        urllib.request.urlretrieve("https://dot.net/v1/dotnet-install.sh", script)
        run(["bash", script, "--channel", "8.0", "--install-dir", home, "--no-path"])
    return exe


def step_launcher():
    """NAM Trainer (the program with the icon in the notification area / menu bar): one C# project for both systems."""
    rid = "win-x64" if TARGET == "windows" else "osx-arm64"
    out = CACHE / "launcher" / rid
    env = {**os.environ, "DOTNET_CLI_HOME": str(CACHE / "dotnet-home"), "NUGET_PACKAGES": str(CACHE / "nuget"), "DOTNET_NOLOGO": "1",
           "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1"}
    run([dotnet(), "publish", HERE / "launcher" / "NamTrainer", "-c", "Release", "-r", rid, "--self-contained", "true",
         "-p:PublishSingleFile=true", "-p:IncludeNativeLibrariesForSelfExtract=true", "-o", out], env=env)
    if TARGET == "windows":
        (STAGE / "trainer").mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "NamTrainer.exe", STAGE / "trainer" / "NAM Trainer.exe")
        say(f"NAM Trainer.exe: {(STAGE / 'trainer' / 'NAM Trainer.exe').stat().st_size / 1e6:.0f} MB")


STEPS = {"tools": step_tools, "launcher": step_launcher, "runtime": step_runtime, "trainer": step_trainer, "variants": step_variants, "capture": step_capture,
         "extras": step_extras, "iss": step_iss}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", default="all", choices=["all", *STEPS])
    ap.add_argument("--target", choices=("windows", "macos"), help="what to build (default: the system this runs on; macOS is built on a Mac, see suite/macos)")
    ap.add_argument("--variants", help="comma list (nvidia,amd,cpu): limit the variants built/packed")
    ap.add_argument("--test-data", help="test builds only: folder the uninstaller treats as NAM Capture's data folder")
    opts = ap.parse_args()
    if opts.target:
        TARGET = opts.target
    if TARGET == "macos" and opts.step not in ("launcher",):
        raise SystemExit("on macOS only the launcher is built here; the package comes from suite/macos/build_mac.sh")
    only = opts.variants.split(",") if opts.variants else None
    if opts.step == "all":
        for name in ("tools", "launcher", "runtime", "trainer", "variants", "capture", "extras", "iss"):
            STEPS[name](only) if name in ("variants", "iss") else STEPS[name]()
    elif opts.step == "iss":
        step_iss(only, opts.test_data)
    elif opts.step == "variants":
        step_variants(only)
    else:
        STEPS[opts.step]()
