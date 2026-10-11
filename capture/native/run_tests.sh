#!/bin/bash
# Compiles and runs the native engine's own tests (native/engine/tests/). No framework: plain asserts, run as a
# normal program - matching tests/test_capture_generators.py's PassthroughGenTests, which this mirrors.
#
# test_rtengine.cpp only sees RTEngine.h's plain C API (opaque struct pointer, no NAM/Eigen/FFTConvolver types
# leak through it), so it links against the already-built librtengine.a (build.sh) instead of recompiling the
# vendored libraries itself.
set -euo pipefail
cd "$(dirname "$0")"
bash build.sh
mkdir -p build
clang++ -std=c++20 -stdlib=libc++ -O0 -g -Wall -Wextra -arch arm64 -Iengine \
  engine/tests/test_rtengine.cpp build/librtengine.a -o build/test_rtengine
./build/test_rtengine

# the tuner's pitch detector is a header (engine/Tuner.h): no library to link
clang++ -std=c++20 -stdlib=libc++ -O2 -Wall -Wextra -arch arm64 -Iengine engine/tests/test_tuner.cpp -o build/test_tuner
./build/test_tuner
