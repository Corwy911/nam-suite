#!/bin/bash
# Builds the macOS parts of NAM Suite (Apple Silicon) into suite/cache/mac/stage and, in the end, the installer package.
#
#   bash suite/macos/build_mac.sh capture     NAM Capture.app   (Swift shell + engine + pinned Python + packages)
#   bash suite/macos/build_mac.sh trainer     NAM Trainer.app   (C# launcher + pinned Python + PyTorch/Metal + trainer)
#   bash suite/macos/build_mac.sh pkg         the two .pkg + NAM-Suite-macOS.pkg (choices: Trainer / Capture / both)
#   bash suite/macos/build_mac.sh all
#
# Everything is built under suite/cache and suite/output (git-ignored); nothing is installed on this Mac. The shared
# steps (pinned Python, the .NET launcher) are the ones of suite/build.py - this script only assembles.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
SUITE="$ROOT/suite"
MAC="$SUITE/cache/mac"
STAGE="$MAC/stage"
VERSION="$(tr -d '[:space:]' < "$SUITE/VERSION")"
export NAM_SUITE_VERSION="$VERSION"
# where the installer package lands: suite/output by default, or NAM_SUITE_OUT (the same env var
# build-installer.cmd sets for suite/build.py on Windows, so both systems can ship installers next to the repo)
OUT="${NAM_SUITE_OUT:-$SUITE/output}"
step="${1:-all}"

say() { echo "[mac] $*"; }

pinned_python() {
  python3 -c "import sys; sys.path.insert(0, '$SUITE'); import build; print(build.standalone_python('macos'))" 2>/dev/null | tail -1
}

# every user may read and enter the program folders (the files of this repository can be owner-only, the installer keeps what it is given),
# then the ad-hoc signature - anything changed after it would invalidate it
sign() { chmod -R u+rwX,go+rX "$1"; codesign --force --deep -s - "$1" >/dev/null 2>&1 && say "signed (ad hoc): $(basename "$1")"; }

step_capture() {
  local py; py="$(pinned_python)"
  local site="$MAC/capture-site"
  say "capture: packages for the pinned Python ($("$py/bin/python3" --version))"
  rm -rf "$site"; mkdir -p "$site"
  "$py/bin/python3" -m pip install --disable-pip-version-check --no-compile --target "$site" -r "$SUITE/installer/requirements-capture.lock" >/dev/null
  rm -rf "$site/bin"
  "$py/bin/python3" -m compileall -q -j 0 --invalidation-mode unchecked-hash "$site"
  mkdir -p "$STAGE"
  NAM_APP_OUT="$STAGE/NAM Capture.app" NAM_SUITE_PYTHON="$py" NAM_SUITE_SITE="$site" bash "$ROOT/capture/packaging/build_app.sh" >/dev/null
  mkdir -p "$STAGE/NAM Capture.app/Contents/Resources/docs"
  cp "$ROOT"/capture/*.pdf "$STAGE/NAM Capture.app/Contents/Resources/docs/" 2>/dev/null || true
  sign "$STAGE/NAM Capture.app"
  say "NAM Capture.app: $(du -sh "$STAGE/NAM Capture.app" | cut -f1)"
}

step_trainer() {
  local py; py="$(pinned_python)"
  local app="$STAGE/NAM Trainer.app" res="$STAGE/NAM Trainer.app/Contents/Resources"
  say "trainer: launcher (C#, osx-arm64)"
  [ -x "$SUITE/cache/launcher/osx-arm64/NamTrainer" ] || python3 "$SUITE/build.py" launcher --target macos >/dev/null
  rm -rf "$app"; mkdir -p "$app/Contents/MacOS" "$res/trainer"
  cp -R "$SUITE/cache/launcher/osx-arm64/." "$app/Contents/MacOS/"
  say "trainer: Python + packages (PyTorch with Metal)"
  rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude 'test' --exclude 'idlelib' --exclude 'turtledemo' --exclude 'ensurepip' \
    --exclude '/lib/python*/site-packages' "$py/" "$res/python/"
  mkdir -p "$res/python/lib/python3.12/site-packages" "$res/trainer/site"
  "$py/bin/python3" -m pip install --disable-pip-version-check --no-compile --target "$res/trainer/site" -r "$SUITE/locks/requirements-macos.lock" >/dev/null
  rm -rf "$res/trainer/site/bin"
  "$py/bin/python3" -m compileall -q -j 0 --invalidation-mode unchecked-hash "$res/trainer/site"
  rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude 'tmp' --exclude 'cache' --exclude 'data' --exclude '*.log' --exclude 'settings.json' \
    --exclude 'hardware.json' --exclude 'start.ps1' --exclude 'stop.ps1' --exclude 'calibration.json' "$ROOT/trainer/nam-web/" "$res/trainer/nam-web/"
  rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ROOT/trainer/trainers/" "$res/trainer/trainers/"
  [ -f "$ROOT/trainer/LICENSE" ] && cp "$ROOT/trainer/LICENSE" "$res/trainer/"
  # icon: suite/assets/nam-trainer.png -> .icns
  local iconset="$MAC/trainer.iconset"; rm -rf "$iconset"; mkdir -p "$iconset"
  for n in 16 32 128 256 512; do
    sips -z $n $n "$SUITE/assets/nam-trainer.png" --out "$iconset/icon_${n}x${n}.png" >/dev/null
    sips -z $((n*2)) $((n*2)) "$SUITE/assets/nam-trainer.png" --out "$iconset/icon_${n}x${n}@2x.png" >/dev/null
  done
  iconutil -c icns "$iconset" -o "$res/AppIcon.icns"
  cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>NAM Trainer</string>
<key>CFBundleDisplayName</key><string>NAM Trainer</string>
<key>CFBundleExecutable</key><string>NamTrainer</string>
<key>CFBundleIconFile</key><string>AppIcon.icns</string>
<key>CFBundleIdentifier</key><string>com.namsuite.trainer</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>$VERSION</string>
<key>CFBundleVersion</key><string>$VERSION</string>
<key>LSMinimumSystemVersion</key><string>13.0</string>
<key>LSUIElement</key><true/>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
  sign "$app"
  say "NAM Trainer.app: $(du -sh "$app" | cut -f1)"
}

step_uninstaller() {
  local app="$STAGE/Disinstalla NAM Suite.app"
  rm -rf "$app"; mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
  cp "$SUITE/macos/uninstall/Disinstalla" "$app/Contents/MacOS/Disinstalla"; chmod +x "$app/Contents/MacOS/Disinstalla"
  cp "$ROOT/capture/packaging/AppIcon.icns" "$app/Contents/Resources/AppIcon.icns"
  cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>Disinstalla NAM Suite</string>
<key>CFBundleExecutable</key><string>Disinstalla</string>
<key>CFBundleIconFile</key><string>AppIcon.icns</string>
<key>CFBundleIdentifier</key><string>com.namsuite.uninstall</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>$VERSION</string>
<key>LSMinimumSystemVersion</key><string>13.0</string>
</dict></plist>
PLIST
  sign "$app"
}

# one component package per piece, then the product with the choices Trainer / Capture (the uninstaller always comes along)
component() {   # <identifier> <app in stage> <scripts folder or ''>
  local id="$1" app="$2" scripts="$3" root="$MAC/pkgroot-$1"
  rm -rf "$root"; mkdir -p "$root"; cp -R "$STAGE/$app" "$root/"
  pkgbuild --analyze --root "$root" "$MAC/$id.plist" >/dev/null
  for i in $(seq 0 8); do /usr/libexec/PlistBuddy -c "Set :$i:BundleIsRelocatable false" "$MAC/$id.plist" 2>/dev/null || true; done
  local args=(--root "$root" --component-plist "$MAC/$id.plist" --identifier "$id" --version "$VERSION" --install-location "/Applications/NAM Suite")
  [ -n "$scripts" ] && { chmod +x "$scripts"/*; args+=(--scripts "$scripts"); }      # git may not keep the executable bit; pkgbuild skips a script that is not executable
  pkgbuild "${args[@]}" "$MAC/pkgs/$id.pkg" >/dev/null
  say "$id.pkg: $(du -sh "$MAC/pkgs/$id.pkg" | cut -f1)"
}

step_pkg() {
  rm -rf "$MAC/pkgs"; mkdir -p "$MAC/pkgs" "$OUT"
  for a in "NAM Capture" "NAM Trainer"; do [ -d "$STAGE/$a.app" ] && sign "$STAGE/$a.app"; done
  step_uninstaller
  component com.namsuite.common "Disinstalla NAM Suite.app" ""
  local choices="" lines="" outline=""
  if [ -d "$STAGE/NAM Trainer.app" ]; then
    component com.namsuite.trainer "NAM Trainer.app" "$SUITE/macos/scripts/trainer"
    choices+='<choice id="trainer" title="NAM Trainer" description="Trains models on the Apple GPU (Metal) / Addestra i modelli sulla GPU Apple (Metal)"><pkg-ref id="com.namsuite.trainer"/></choice>'
    lines+='<pkg-ref id="com.namsuite.trainer" version="'$VERSION'">com.namsuite.trainer.pkg</pkg-ref>'
    outline+='<line choice="trainer"/>'
  fi
  if [ -d "$STAGE/NAM Capture.app" ]; then
    component com.namsuite.capture "NAM Capture.app" "$SUITE/macos/scripts/capture"
    choices+='<choice id="capture" title="NAM Capture" description="Calibrated reamp captures / Catture reamp calibrate"><pkg-ref id="com.namsuite.capture"/></choice>'
    lines+='<pkg-ref id="com.namsuite.capture" version="'$VERSION'">com.namsuite.capture.pkg</pkg-ref>'
    outline+='<line choice="capture"/>'
  fi
  cat > "$MAC/distribution.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>NAM Suite $VERSION</title>
  <welcome file="Welcome.html"/>
  <readme file="ReadMe.html"/>
  <options customize="always" require-scripts="false" hostArchitectures="arm64"/>
  <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
  <choices-outline>$outline<line choice="common"/></choices-outline>
  $choices
  <choice id="common" visible="false" start_selected="true"><pkg-ref id="com.namsuite.common"/></choice>
  $lines
  <pkg-ref id="com.namsuite.common" version="$VERSION">com.namsuite.common.pkg</pkg-ref>
</installer-gui-script>
XML
  productbuild --distribution "$MAC/distribution.xml" --resources "$SUITE/macos/resources" --package-path "$MAC/pkgs" "$OUT/NAM-Suite-macOS.pkg" >/dev/null
  (cd "$OUT" && shasum -a 256 NAM-Suite-macOS.pkg > NAM-Suite-macOS.pkg.sha256)
  say "NAM-Suite-macOS.pkg: $(du -sh "$OUT/NAM-Suite-macOS.pkg" | cut -f1)"
}

case "$step" in
  capture) step_capture ;;
  trainer) step_trainer ;;
  pkg) step_pkg ;;
  all) step_capture; step_trainer; step_pkg ;;
  *) echo "usage: build_mac.sh capture|trainer|pkg|all"; exit 1 ;;
esac
