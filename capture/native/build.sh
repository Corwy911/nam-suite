#!/bin/bash
# Builds the native real-time engine as a static library (native/build/librtengine.a, arm64) with plain clang++ -
# no CMake, no external build system: this project deliberately keeps only what it needs (see HANDOFF-MAC.md /
# CLAUDE.md's "keep it simple" thread running through setup-mac.sh, build_app.sh, and this file). NeuralAmpModelerCore
# and FFTConvolver are vendored into native/third_party/ (see third_party/VENDORED.md for exact commits/licenses) -
# copied source, not a submodule/FetchContent, same reasoning as everything else in this file.
#
# -std=c++20: NeuralAmpModelerCore itself needs it (std::atomic<std::shared_ptr<T>>, used both by its own
# wavenet/slimmable.h and by our own RTEngine.cpp for the model/IR hot-swap - see RTEngine.cpp's own comment).
# -DNAM_SAMPLE_FLOAT: makes NAM's internal sample type float instead of the double default, matching RTEngine's
# float buffers end to end (no per-block double<->float conversion needed).
set -euo pipefail
cd "$(dirname "$0")"

BUILD="build"
mkdir -p "$BUILD"

THIRD_PARTY="third_party"
NAM_SOURCES=("$THIRD_PARTY"/nam/NAM/*.cpp "$THIRD_PARTY"/nam/NAM/wavenet/*.cpp)
FFTCONVOLVER_SOURCES=("$THIRD_PARTY"/fftconvolver/*.cpp)
SOURCES=(engine/RTEngine.cpp "${NAM_SOURCES[@]}" "${FFTCONVOLVER_SOURCES[@]}")
INCLUDES=(-Iengine -I"$THIRD_PARTY/nam" -I"$THIRD_PARTY/nam/nlohmann" -I"$THIRD_PARTY/eigen" -I"$THIRD_PARTY/fftconvolver")
DEFINES=(-DNAM_SAMPLE_FLOAT)

echo "== compiling the engine (arm64) =="
OBJS=()
for src in "${SOURCES[@]}"; do
  obj="$BUILD/$(basename "${src%.cpp}").o"
  # the vendored libraries build with a handful of harmless -Wunused-parameter warnings upstream; only our own
  # RTEngine.cpp is held to -Werror-like scrutiny (reviewed by hand, not enforced here to keep this script simple)
  clang++ -std=c++20 -stdlib=libc++ -O2 -Wall -Wextra -arch arm64 "${DEFINES[@]}" "${INCLUDES[@]}" -c "$src" -o "$obj"
  OBJS+=("$obj")
done

echo "== archiving librtengine.a =="
ar rcs "$BUILD/librtengine.a" "${OBJS[@]}"

echo "done: $BUILD/librtengine.a"
