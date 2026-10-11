@echo off
rem Same as build_controlserver.bat, but with NAM_HAVE_ASIO_SDK defined and the ASIO SDK's own host-side sources
rem compiled in: this is the real Phase 4 build, once native\third_party\asio_sdk\ has been dropped in (see
rem native\third_party\ASIO_SDK_SETUP.md). Only the host pieces of the SDK are needed (not the driver-side
rem helpers like combase.cpp/register.cpp/dllentry.cpp, which are for writing a driver, not hosting one).
setlocal
cd /d "%~dp0"

set NATIVE=..\..\native
set ASIO_SDK=%NATIVE%\third_party\asio_sdk
if not exist "%NATIVE%\build\rtengine.lib" (
    echo native\build\rtengine.lib non trovato: esegui prima native\build.bat
    exit /b 1
)
if not exist "%ASIO_SDK%\common\asio.h" (
    echo SDK ASIO non trovato in %ASIO_SDK%: vedi native\third_party\ASIO_SDK_SETUP.md
    exit /b 1
)

if not exist build mkdir build
rem /WHOLEARCHIVE:rtengine.lib (linker flag, after /link): NAM's architecture parsers (WaveNet, ConvNet, ...)
rem self-register via static initializers with nothing else in the program referencing their .obj directly, so
rem the linker drops them from a static .lib unless forced to keep the whole archive - exactly the same reason
rem Mac's build_app.sh needs -force_load. Found by an actual "No config parser registered for architecture:
rem WaveNet" error loading a real model, not by inspection - the native test suite only tests a MISSING model
rem file, never a real load, so this gap was invisible until tested against one.
cl.exe /std:c++20 /O2 /W3 /EHsc /MD /nologo /D_USE_MATH_DEFINES /DNAM_HAVE_ASIO_SDK ^
  /I"%NATIVE%\engine" /I"%ASIO_SDK%\common" /I"%ASIO_SDK%\host" /I"%ASIO_SDK%\host\pc" ^
  AudioEngineWin.cpp ControlServerWin.cpp main_controlserver.cpp ^
  "%ASIO_SDK%\common\asio.cpp" "%ASIO_SDK%\host\asiodrivers.cpp" "%ASIO_SDK%\host\pc\asiolist.cpp" ^
  "%NATIVE%\build\rtengine.lib" ws2_32.lib ole32.lib advapi32.lib user32.lib ^
  /Fe:build\controlserver_asio.exe /Fo:build\ ^
  /link /WHOLEARCHIVE:"%NATIVE%\build\rtengine.lib"
if errorlevel 1 exit /b 1

echo done: build\controlserver_asio.exe
endlocal
