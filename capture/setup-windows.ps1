# Sets up NAM Capture on Windows: a virtual environment INSIDE this folder (nothing is installed system-wide).
# Equivalent of setup-mac.sh. Requires Python 3.10+ (get.python.org) with the launcher on PATH.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$found = Get-Command $py -ErrorAction SilentlyContinue
if (-not $found) {
    Write-Host "$py non trovato. Installa Python 3.10+ da https://www.python.org/downloads/windows/ e rilancia."
    exit 1
}
& $py -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Serve Python 3.10 o piu' recente."
    exit 1
}

& $py -m venv venv
& venv\Scripts\python.exe -m pip install --upgrade pip
# httpx (not the Mac build's renamed "httpx2") - only fastapi's TestClient (tests/) needs it; excluded from the
# packaged app by name instead, same end result as the Mac exclusion without the rename (see packaging/build_app.ps1).
& venv\Scripts\python.exe -m pip install fastapi "uvicorn[standard]" pydantic python-multipart packaging numpy soundfile sounddevice httpx reportlab pypdfium2

Write-Host ""
Write-Host "Fatto. Prova:"
Write-Host "  venv\Scripts\python.exe -m unittest discover -s tests"
Write-Host "  venv\Scripts\python.exe capture_tool\app.py --port 8766   (poi apri http://localhost:8766)"
Write-Host ""
Write-Host "Per compilare l'app Windows vera e propria: vedi packaging\ (non ancora pronto)."
