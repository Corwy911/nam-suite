@echo off
rem Builds the standalone ControlServerWin console harness (main_controlserver.cpp) for Phase 3/testing: no ASIO SDK
rem needed (NAM_HAVE_ASIO_SDK left undefined -> AudioEngineWin's open()/open_dual() cleanly fail with a "not
rem available" error instead of touching hardware), so this proves the HTTP route contract and RTEngine wiring
rem independently of ASIO and of the WinUI3 shell. Run from a "x64 Native Tools Command Prompt for VS 2022", after
rem native\build.bat has produced native\build\rtengine.lib.
setlocal
cd /d "%~dp0"

set NATIVE=..\..\native
if not exist "%NATIVE%\build\rtengine.lib" (
    echo native\build\rtengine.lib non trovato: esegui prima native\build.bat
    exit /b 1
)

if not exist build mkdir build
cl.exe /std:c++20 /O2 /W3 /EHsc /MD /nologo /D_USE_MATH_DEFINES ^
  /I"%NATIVE%\engine" ^
  AudioEngineWin.cpp ControlServerWin.cpp main_controlserver.cpp "%NATIVE%\build\rtengine.lib" ws2_32.lib ole32.lib ^
  /Fe:build\controlserver_test.exe /Fo:build\
if errorlevel 1 exit /b 1

echo done: build\controlserver_test.exe
endlocal
