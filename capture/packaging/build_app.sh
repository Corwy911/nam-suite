#!/bin/bash
# Builds "NAM Capture.app" next to this script: a real Mac app (own icon, own Dock entry, Cmd+Q, no browser chrome).
# Bundles the project's own venv and source as loose files (not frozen with py2app): the capture engine's own
# subprocess model (capture/session.py spawning capture/engine.py with sys.executable) needs a real, plain Python
# layout to keep working, so this is the sturdier way to package it, not a shortcut.
#
#   bash packaging/build_app.sh
#
# Needs: the project's venv already set up (bash setup-mac.sh), Xcode Command Line Tools (swiftc), macOS's iconutil.
set -euo pipefail
cd "$(dirname "$0")/.."       # project root
ROOT="$(pwd)"
PKG="$ROOT/packaging"
VERSION="$(tr -d '[:space:]' < "$ROOT/../suite/VERSION")"      # the Suite's version (suite/VERSION), one number for every app
APP="$PKG/NAM Capture.app"

if [ ! -x "$ROOT/venv/bin/python" ]; then
  echo "Manca venv/: esegui prima  bash setup-mac.sh"
  exit 1
fi
if ! command -v swiftc >/dev/null 2>&1; then
  echo "Manca swiftc: installa gli Xcode Command Line Tools (xcode-select --install)."
  exit 1
fi

echo "== icona =="
if [ ! -f "$PKG/AppIcon.icns" ]; then
  venv/bin/python "$PKG/make_icon.py"
fi

echo "== pulizia build precedente =="
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

echo "== motore audio nativo (native/) =="
bash "$ROOT/native/build.sh"

echo "== compilazione del launcher nativo (Swift/AppKit/WebKit/CoreAudio) =="
# -force_load: NeuralAmpModelerCore registers each architecture (WaveNet, LSTM, ConvNet, Linear, SlimmableContainer)
# with a self-registering static object in its own .cpp (nam::ConfigParserHelper, see model_config.h) - nothing else
# in this codebase calls into those files directly (lookup happens by name, at runtime, through the registry), so a
# plain link against the static archive lets the linker drop every one of those .o's as "unreferenced", and NAM model
# loading fails for every architecture ("No config parser registered for..."). force_load keeps the whole archive,
# static initializers included, regardless of what's referenced.
swiftc "$PKG/main.swift" "$PKG/AudioEngine.swift" "$PKG/ControlServer.swift" \
  -import-objc-header "$PKG/RTEngineBridge.h" \
  -O -o "$APP/Contents/MacOS/NAMCapture" \
  -framework Cocoa -framework WebKit -framework AudioToolbox -framework CoreAudio -framework AVFoundation \
  -Xlinker -force_load -Xlinker "$ROOT/native/build/librtengine.a" -lc++
chmod +x "$APP/Contents/MacOS/NAMCapture"

echo "== Info.plist =="
sed "s/__VERSION__/$VERSION/g" "$PKG/Info.plist.template" > "$APP/Contents/Info.plist"
cp "$PKG/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

echo "== copia del programma (capture_tool/) =="
mkdir -p "$APP/Contents/Resources/app"
rsync -a --exclude '__pycache__' --exclude 'tmp' --exclude '*.pyc' "$ROOT/capture_tool/" "$APP/Contents/Resources/app/"

echo "== copia dell'ambiente Python (venv/, solo quello che serve a runtime) =="
# The venv also holds development-only packages: scipy (nothing imports it), reportlab/pypdfium2/PIL (only manuale/ builds
# the PDFs), pip, and httpx2/httpcore2 (only the tests' TestClient). Left out of the app: it goes from ~150 MB to ~40 MB.
rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '/lib/python*/site-packages/scipy*' --exclude '/lib/python*/site-packages/PIL' --exclude '/lib/python*/site-packages/pillow*' \
  --exclude '/lib/python*/site-packages/reportlab*' --exclude '/lib/python*/site-packages/pypdfium2*' \
  --exclude '/lib/python*/site-packages/pip' --exclude '/lib/python*/site-packages/pip-*' \
  --exclude '/lib/python*/site-packages/httpx2*' --exclude '/lib/python*/site-packages/httpcore2*' \
  "$ROOT/venv/" "$APP/Contents/Resources/venv/"

echo
echo "Fatto: \"$APP\""
echo "Apri con:  open \"$APP\""
echo "(la prima volta macOS può chiedere il permesso di usare il microfono: è la scheda audio da calibrare)"
