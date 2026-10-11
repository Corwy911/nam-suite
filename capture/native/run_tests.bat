@echo off
rem Compiles and runs the native engine's own tests (native\engine\tests\). Windows equivalent of run_tests.sh.
rem Run from a "x64 Native Tools Command Prompt for VS 2022".
setlocal
cd /d "%~dp0"

call ".\build.bat"
if errorlevel 1 exit /b 1

if not exist build mkdir build
cl.exe /std:c++20 /Od /Zi /W3 /EHsc /MD /nologo /D_USE_MATH_DEFINES /Iengine engine\tests\test_rtengine.cpp build\rtengine.lib /Fe:build\test_rtengine.exe /Fo:build\ /Fd:build\test_rtengine.pdb
if errorlevel 1 exit /b 1

.\build\test_rtengine.exe
if errorlevel 1 exit /b 1

cl.exe /std:c++20 /O2 /W3 /EHsc /MD /nologo /D_USE_MATH_DEFINES /Iengine engine\tests\test_tuner.cpp /Fe:build\test_tuner.exe /Fo:build\
if errorlevel 1 exit /b 1
.\build\test_tuner.exe
endlocal
