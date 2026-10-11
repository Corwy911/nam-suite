@echo off
rem NAM Suite - scarica i 4 file dell'installatore dall'ultima release di GitHub, li controlla e avvia l'installazione.
rem NAM Suite - downloads the installer's 4 files from the latest GitHub release, checks them and starts the setup.
rem   Doppio clic / double-click. Mette i file nella cartella "NAM-Suite-Setup" accanto a questo file. Si puo' interrompere e rilanciare: riprende da dove era.
rem   Argomento / argument:  /solo-controllo  (non scarica ne' avvia: stampa solo cosa farebbe / prints only what it would do)
setlocal EnableDelayedExpansion
chcp 65001 >nul
title NAM Suite - download

set "REPO=Corwy911/nam-suite"
set "BASE=https://github.com/%REPO%/releases/latest/download"
if defined NAM_SUITE_BASE set "BASE=%NAM_SUITE_BASE%"          rem (only for testing the script against a local server)
set "DEST=%~dp0NAM-Suite-Setup"
set "DRY=0"
if /i "%~1"=="/solo-controllo" set "DRY=1"

echo.
echo  NAM Suite - download dell'installatore / installer download
echo  Cartella / folder: %DEST%
echo.
if not exist "%DEST%" mkdir "%DEST%"
where curl.exe >nul 2>nul
if errorlevel 1 ( echo  curl.exe non trovato ^(serve Windows 10 o 11^). / curl.exe not found ^(needs Windows 10 or 11^). & goto :fail )

rem 1) the checksums first: they say what to download and how to verify it
curl.exe -fsSL "%BASE%/SHA256SUMS.txt" -o "%DEST%\SHA256SUMS.txt"
if errorlevel 1 ( echo  Non riesco a leggere la release su GitHub ^(rete? release non ancora pubblicata?^). / Cannot read the release on GitHub. & goto :fail )

set "FILES="
for /f "usebackq tokens=1,2 delims=* " %%A in ("%DEST%\SHA256SUMS.txt") do set "FILES=!FILES! %%B"
if "!FILES!"=="" ( echo  SHA256SUMS.txt vuoto. / SHA256SUMS.txt is empty. & goto :fail )
if "%DRY%"=="1" ( echo  File che scaricherei / files I would download:!FILES! & goto :eof )

rem 2) each file: skipped when it is already complete and correct, else downloaded (resuming a partial one)
for %%F in (!FILES!) do (
    call :verify "%%F"
    if errorlevel 1 (
        echo  Scarico / downloading %%F ...
        curl.exe -fL --retry 5 --retry-delay 3 -C - --progress-bar "%BASE%/%%F" -o "%DEST%\%%F"
        if errorlevel 1 ( echo  Download di %%F non riuscito. Rilancia questo file per riprendere. / Download of %%F failed. Run this file again to resume. & goto :fail )
        call :verify "%%F"
        if errorlevel 1 ( echo  %%F scaricato ma il controllo non torna: cancello e rilancia. / %%F downloaded but its check fails: deleting it, run this file again. & del "%DEST%\%%F" & goto :fail )
    ) else (
        echo  Gia' presente e corretto / already there and correct: %%F
    )
)

echo.
echo  Tutto scaricato e verificato. Avvio l'installazione... / All downloaded and verified. Starting the setup...
start "" "%DEST%\NAM-Suite-Setup.exe"
goto :eof

:verify
rem %1 = file name; exit 0 when "%DEST%\file" exists and its SHA-256 equals the one in SHA256SUMS.txt
if not exist "%DEST%\%~1" exit /b 1
set "WANT="
for /f "usebackq tokens=1,2 delims=* " %%A in ("%DEST%\SHA256SUMS.txt") do if /i "%%B"=="%~1" set "WANT=%%A"
set "GOT="
for /f "skip=1 tokens=* delims=" %%H in ('certutil -hashfile "%DEST%\%~1" SHA256 ^| findstr /v /c:"CertUtil"') do if not defined GOT set "GOT=%%H"
set "GOT=!GOT: =!"
if /i "!GOT!"=="!WANT!" ( exit /b 0 ) else ( exit /b 1 )

:fail
echo.
pause
exit /b 1
