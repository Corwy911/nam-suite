@echo off
rem Builds the native real-time engine as a static library (native\build\rtengine.lib, x64) with MSVC cl.exe.
rem Windows equivalent of build.sh: same sources, same -DNAM_SAMPLE_FLOAT/-std:c++20, no -stdlib=libc++/-arch
rem (Apple/clang-specific, simply dropped). Must run from a "x64 Native Tools Command Prompt for VS 2022" (or
rem after calling vcvarsall.bat x64) so cl.exe/lib.exe are on PATH.
setlocal enabledelayedexpansion
cd /d "%~dp0"

where cl.exe >nul 2>nul
if errorlevel 1 (
    echo cl.exe non trovato sul PATH. Esegui questo script da una "x64 Native Tools Command Prompt for VS 2022".
    exit /b 1
)

set BUILD=build
if not exist "%BUILD%" mkdir "%BUILD%"

set THIRD_PARTY=third_party
set INCLUDES=/Iengine /I"%THIRD_PARTY%\nam" /I"%THIRD_PARTY%\nam\nlohmann" /I"%THIRD_PARTY%\eigen" /I"%THIRD_PARTY%\fftconvolver"
set DEFINES=/DNAM_SAMPLE_FLOAT /D_USE_MATH_DEFINES
rem /MD (dynamic release CRT): must match whatever links this .lib later (NamCaptureApp, the test harnesses) -
rem a mismatched /MT vs /MD vs /MDd here is a linker error (LNK2038), not a warning, if ever changed.
set CFLAGS=/std:c++20 /O2 /W3 /EHsc /MD /nologo /c

echo == compiling the engine (x64) ==
set OBJS=
for %%f in (engine\RTEngine.cpp %THIRD_PARTY%\nam\NAM\*.cpp %THIRD_PARTY%\nam\NAM\wavenet\*.cpp %THIRD_PARTY%\fftconvolver\*.cpp) do (
    echo   %%f
    cl.exe %CFLAGS% %DEFINES% %INCLUDES% "%%f" /Fo"%BUILD%\%%~nf.obj"
    if errorlevel 1 exit /b 1
    set OBJS=!OBJS! "%BUILD%\%%~nf.obj"
)

echo == archiving rtengine.lib ==
lib.exe /nologo /out:"%BUILD%\rtengine.lib" !OBJS!
if errorlevel 1 exit /b 1

echo done: %BUILD%\rtengine.lib
endlocal
