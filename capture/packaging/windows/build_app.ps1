# Windows equivalent of packaging/build_app.sh: builds the native engine, the shell exe (NamCaptureApp), and
# assembles a distributable folder (packaging\windows\output\NAM Capture\) with the exe, capture_tool\ (trimmed)
# and a trimmed venv\ - mirrors the Mac script's own exclusion list (scipy, Pillow, reportlab, pypdfium2, pip: all
# dev/test-only, see setup-windows.ps1's own comment on httpx/httpcore replacing the Mac build's renamed httpx2).
#
# Run from an ordinary PowerShell prompt (not a VS developer prompt): this script finds vcvarsall.bat itself via
# vswhere, the same way a real release build should not depend on the operator's shell already being set up.
$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\..\.."          # nam-capture/
$Native = Join-Path $Root "native"
$WindowsDir = Join-Path $Root "packaging\windows"
$AppProj = Join-Path $WindowsDir "NamCaptureApp\NamCaptureApp.vcxproj"
$Output = Join-Path $WindowsDir "output\NAM Capture"

function Get-VsDevCmdVars {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe non trovato: serve Visual Studio/Build Tools 2022 con il carico C++." }
    $vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) { throw "Nessuna installazione di Visual Studio con il carico C++ trovata." }
    return Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"
}

# The Suite's version (suite/VERSION), the same one build_app.sh reads
$Version = (Get-Content (Join-Path $Root "..\suite\VERSION") -Raw).Trim()
Write-Host "NAM Capture $Version (Windows)"

Write-Host "== native engine =="
$vcvarsall = Get-VsDevCmdVars
$buildBat = Join-Path $Native "build.bat"
cmd /c "`"$vcvarsall`" x64 && `"$buildBat`""
if ($LASTEXITCODE -ne 0) { throw "native\build.bat fallito" }

Write-Host "== shell (NamCaptureApp.exe) =="
$msbuild = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\amd64\MSBuild.exe"
if (-not (Test-Path $msbuild)) { throw "MSBuild.exe non trovato" }
& $msbuild $AppProj /t:build /p:Configuration=Release /p:Platform=x64 /nologo /v:minimal
if ($LASTEXITCODE -ne 0) { throw "build di NamCaptureApp.exe fallita" }

Write-Host "== assemblaggio in $Output =="
if (Test-Path $Output) { Remove-Item -Recurse -Force $Output }
New-Item -ItemType Directory -Force -Path $Output | Out-Null

Copy-Item (Join-Path $WindowsDir "NamCaptureApp\x64\Release\NamCaptureApp.exe") $Output
Copy-Item (Join-Path $WindowsDir "NamCaptureApp\x64\Release\WebView2Loader.dll") $Output

# capture_tool/: everything except runtime junk (same spirit as build_app.sh's rsync excludes)
$captureToolDest = Join-Path $Output "capture_tool"
robocopy (Join-Path $Root "capture_tool") $captureToolDest /E /XD "__pycache__" "tmp" /XF "settings.json" *.pyc | Out-Null

# venv/: exclude dev/test-only packages (scipy, Pillow, reportlab, pypdfium2, pip, httpx/httpcore - see
# setup-windows.ps1's comment: these replace the Mac build's renamed httpx2/httpcore2 trick with plain exclusion)
$venvDest = Join-Path $Output "venv"
robocopy (Join-Path $Root "venv") $venvDest /E /XD "__pycache__" "scipy" "PIL" "Pillow" "reportlab" "pypdfium2" "pip" "httpx" "httpcore" /XF *.pyc | Out-Null

Write-Host "fatto: $Output"
