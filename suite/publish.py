"""Prepares what goes to GitHub: a clean copy of the repository's tracked files, WITHOUT what stays private (the manuals, notes for Claude, the Mac hand-over).

The working repository keeps the whole story and everything (the manual is written and kept up to date elsewhere); the public repository gets one
commit per release, made from a snapshot. Usage:

    python suite/publish.py export <folder>      copies the tracked files minus PUBLISH_EXCLUDE into <folder> (its .git is left alone, files that no longer
                                                 exist are removed), ready for `git add -A && git commit` there
    python suite/publish.py list                 what would be left out
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLISH_EXCLUDE = (
    "CLAUDE.md",                                   # notes for the AI assistant that works on the project
    "docs/",                                       # internal plans (the Mac port)
    "capture/manuale/",                            # the manual's sources and its builder
    "capture/Manuale-NAM-Capture-IT.pdf",
    "capture/NAM-Capture-Manual-EN.pdf",
    "capture/OBIETTIVO-UNITY-GAIN.md",             # a design note
    "suite/macos/",                                # the Mac hand-over (comes with the Mac release)
)


def tracked() -> list:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def excluded(path: str) -> bool:
    return any(path == e or (e.endswith("/") and path.startswith(e)) for e in PUBLISH_EXCLUDE)


def export(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    keep = [p for p in tracked() if not excluded(p) and (ROOT / p).is_file()]
    wanted = set(keep)
    for p in keep:
        target = dest / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / p, target)
    for f in sorted(dest.rglob("*"), key=lambda x: len(x.parts), reverse=True):      # remove what the repository no longer has (never .git)
        rel = f.relative_to(dest).as_posix()
        if rel == ".git" or rel.startswith(".git/"):
            continue
        if f.is_file() and rel not in wanted:
            f.unlink()
        elif f.is_dir() and not any(f.iterdir()):
            f.rmdir()
    return len(keep)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "export" and len(sys.argv) == 3:
        print(f"exported {export(Path(sys.argv[2]))} files to {sys.argv[2]}")
    elif cmd == "list":
        print("\n".join(p for p in tracked() if excluded(p)))
    else:
        print(__doc__)
        sys.exit(1)
