# Steinberg ASIO SDK - not vendored here

Unlike NeuralAmpModelerCore, FFTConvolver and Eigen (see `VENDORED.md`), the Steinberg ASIO SDK is **not** copied
into this repository, and its folder is in `.gitignore`. Steinberg's license lets you distribute the *binary* you
build with the SDK, but not the SDK's own headers/source - so it must not be committed, even to a private,
no-remote repo (easier to keep out from day one than to scrub out of history later).

## Getting the SDK

1. Download it from Steinberg's developer site (registration required - a few minutes, free):
   https://www.steinberg.net/developers/ - "3rd Party Developer" -> ASIO SDK. This step needs a human: it asks for
   an email/name and to accept the SDK license, which nobody should do on your behalf.
2. Unzip it so the layout looks like this (the exact folder names the build scripts expect):

   ```
   native/third_party/asio_sdk/
     common/        asio.h, asiosys.h, asiodrivers.h, iasiodrv.h, ...
     host/          asiodrivers.cpp, ...
     host/pc/       asiolist.cpp, ...
   ```

   (This is the SDK's own top-level layout, just dropped in under `asio_sdk/` - no renaming needed.)
3. Build with `NAM_HAVE_ASIO_SDK` defined (see `packaging/windows/build_controlserver.bat` and the real app's
   build script once it exists) - without it, `AudioEngineWin::open()`/`open_dual()` compile as a harmless stub that
   fails with "ASIO SDK non disponibile in questa build" instead of refusing to build.

## Why this shape

`packaging/windows/AudioEngineWin.cpp` (see its own top comment) is written against the SDK's public, stable
host-side C API (`ASIOInit`, `ASIOCreateBuffers`, the `bufferSwitch` callback, ...) exactly as any ASIO host does -
it was written before the SDK was available on this machine, so treat the first real build against it as the actual
verification of that file, not this note.
