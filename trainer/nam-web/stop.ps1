# Completely stops the NAM web server and any training it is running, for THIS installation only.
# (Another NAM folder on the same PC, or any other program using the port, is never touched.)
$port = 8765
# Language of the messages: the one chosen when installing (nam-web\hardware.json, "language"); Italian if unknown.
$lang = 'it'
try {
    $hw = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'hardware.json') -Raw -ErrorAction Stop | ConvertFrom-Json
    if ($hw.language -eq 'en') { $lang = 'en' }
} catch { }
function T([string]$it, [string]$en) { if ($lang -eq 'en') { return $en } else { return $it } }

$root = Split-Path -Parent $PSScriptRoot   # the NAM folder this script belongs to
$targets = @{}

function Test-Mine($process) {
    # A process belongs to this installation if its program, its command line or its parent (the venv launcher)
    # is inside $root.
    if (-not $process) { return $false }
    if ($process.ExecutablePath -and $process.ExecutablePath.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    if ($process.CommandLine -and $process.CommandLine.IndexOf($root, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $true }
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.ParentProcessId)" -ErrorAction SilentlyContinue
    return [bool]($parent -and $parent.ExecutablePath -and $parent.ExecutablePath.StartsWith($root, [StringComparison]::OrdinalIgnoreCase))
}

# The server, its launcher, any training worker and any trainer self-test (updates) of this installation.
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -and ($_.CommandLine -like "*$PSScriptRoot\app.py*" -or $_.CommandLine -like "*$PSScriptRoot\worker.py*" -or $_.CommandLine -like "*$PSScriptRoot\selftest.py*") } |
    ForEach-Object { $targets[[int]$_.ProcessId] = $_.Name }

# Fallback: a python process of this installation still listening on the port (e.g. a server started by hand).
$foreign = $false
Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)" -ErrorAction SilentlyContinue
    if ($proc -and $proc.Name -like 'python*' -and (Test-Mine $proc)) { $targets[[int]$proc.ProcessId] = $proc.Name }
    elseif ($proc -and -not $targets.ContainsKey([int]$proc.ProcessId)) { $foreign = $true }
}

if ($targets.Count -eq 0) {
    if ($foreign) { Write-Host (T "NAM Trainer di questa cartella non era in esecuzione (la porta $port e usata da un altro programma o da un'altra installazione: non lo tocco)." "NAM Trainer of this folder was not running (port $port is used by another program or another installation: I do not touch it).") }
    else { Write-Host (T 'NAM Trainer non era in esecuzione.' 'NAM Trainer was not running.') }
    exit 0
}

foreach ($id in $targets.Keys) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }

$deadline = (Get-Date).AddSeconds(15)
while ((Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 300
}
$still = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($still) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($still[0].OwningProcess)" -ErrorAction SilentlyContinue
    if ($owner -and (Test-Mine $owner)) {
        Write-Host (T "ATTENZIONE: la porta $port e ancora occupata." "WARNING: port $port is still in use.")
        exit 1
    }
}
Write-Host (T "NAM Trainer fermato ($($targets.Count) processi terminati). La GPU e libera." "NAM Trainer stopped ($($targets.Count) processes ended). The GPU is free.")
