#!/bin/bash
# Sets up NAM Capture on macOS: a virtual environment INSIDE this folder (nothing is installed system-wide).
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 non trovato. Installa Python 3.10+ da https://www.python.org/downloads/macos/ (o: brew install python) e rilancia."
  exit 1
fi
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || { echo "Serve Python 3.10 o più recente."; exit 1; }

"$PY" -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install fastapi "uvicorn[standard]" pydantic python-multipart packaging numpy soundfile sounddevice httpx2 reportlab pypdfium2

echo
echo "Fatto. Prova:"
echo "  venv/bin/python -m unittest discover -s tests"
echo "  venv/bin/python capture_tool/app.py --port 8766   (poi apri http://localhost:8766)"
echo
echo "Per compilare l'app Mac vera e propria: bash packaging/build_app.sh"
