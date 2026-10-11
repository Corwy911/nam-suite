# Starts the NAM web server in the background (no console window) and opens the GUI.
param([switch]$NoBrowser)
# Language of the messages: the one chosen when installing (nam-web\hardware.json, "language"); Italian if unknown.
$lang = 'it'
try {
    $hw = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'hardware.json') -Raw -ErrorAction Stop | ConvertFrom-Json
    if ($hw.language -eq 'en') { $lang = 'en' }
} catch { }
function T([string]$it, [string]$en) { if ($lang -eq 'en') { return $en } else { return $it } }


$root = Split-Path -Parent $PSScriptRoot
$port = 8765
$python = Join-Path $root 'venv\Scripts\python.exe'
$app = Join-Path $PSScriptRoot 'app.py'

function Test-Listening {
    [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

function Test-ListenerIsMine {
    # Is the program listening on the port part of THIS installation (not another NAM folder or another program)?
    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    $proc = if ($listener) { Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue }
    if (-not $proc) { return $false }
    if ($proc.ExecutablePath -and $proc.ExecutablePath.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    if ($proc.CommandLine -and $proc.CommandLine.IndexOf($root, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $true }
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.ParentProcessId)" -ErrorAction SilentlyContinue
    return [bool]($parent -and $parent.ExecutablePath -and $parent.ExecutablePath.StartsWith($root, [StringComparison]::OrdinalIgnoreCase))
}

if (Test-Listening) {
    if (-not (Test-ListenerIsMine)) {
        Write-Host (T "La porta $port e gia usata da un altro programma o da un'altra installazione di NAM." "Port $port is already used by another program or another NAM installation.")
        Write-Host (T "Fermalo (con il suo 'Ferma NAM.bat') prima di avviare questa." "Stop it (with its own 'Ferma NAM.bat') before starting this one.")
        exit 1
    }
    Write-Host (T 'NAM Trainer era gia in esecuzione.' 'NAM Trainer was already running.')
} else {
    Write-Host (T 'Avvio NAM Trainer...' 'Starting NAM Trainer...')
    # The full path of app.py in the command line is how stop.ps1 recognises the process.
    Start-Process -FilePath $python -ArgumentList "`"$app`"" -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $PSScriptRoot 'server.out.log') `
        -RedirectStandardError (Join-Path $PSScriptRoot 'server.err.log')
    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-Listening)) {
        if ((Get-Date) -gt $deadline) {
            Write-Host (T 'Il server non e partito, controlla nam-web\server.err.log' 'The server did not start, check nam-web\server.err.log')
            exit 1
        }
        Start-Sleep -Milliseconds 500
    }
}

Write-Host ''
Write-Host ("  " + (T "Su questo PC:  " "On this PC:     ") + "http://localhost:$port")
# The LAN switch lives in the page and is saved in settings.json (default: on).
$lanOn = $true
try {
    $settingsFile = Join-Path $PSScriptRoot 'settings.json'
    if (Test-Path -LiteralPath $settingsFile) {
        $value = (Get-Content -LiteralPath $settingsFile -Raw | ConvertFrom-Json).lan
        if ($null -ne $value) { $lanOn = [bool]$value }
    }
} catch { }
if ($lanOn) {
    Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
        ForEach-Object { Write-Host ("  " + (T "Dalla LAN:     " "From the LAN:   ") + "http://$($_.IPAddress):$port") }
} else {
    Write-Host (T '  LAN disattivata: la pagina e usabile solo da questo PC (interruttore LAN nella pagina).' '  LAN off: the page can only be used from this PC (LAN switch in the page).')
}
Write-Host ''

if (-not $NoBrowser) { Start-Process "http://localhost:$port" }
