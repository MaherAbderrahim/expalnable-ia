param(
    [switch]$NoBrowser,
    [switch]$NoTunnel
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
try { $Host.UI.RawUI.WindowTitle = "Demo IA Prostate - Lanceur" } catch {}

function Step([string]$Text) {
    Write-Host ""
    Write-Host "=== $Text" -ForegroundColor Cyan
}

function Fail([string]$Text) {
    Write-Host ""
    Write-Host "ERREUR : $Text" -ForegroundColor Red
    Read-Host "Appuie sur Entree pour fermer"
    exit 1
}

function Stop-Port([int]$Port) {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $conns) {
        try { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction Stop } catch {}
    }
}

function Wait-Http([string]$Url, [int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { return $true }
        } catch {}
        Start-Sleep -Milliseconds 800
    }
    return $false
}

Write-Host ""
Write-Host "  Demo IA expliquable - cancer de la prostate (Phase 2)" -ForegroundColor Yellow
Write-Host "  -------------------------------------------------------" -ForegroundColor DarkGray

$Python = Join-Path $Root "prostate_model_demo\.venv\Scripts\python.exe"
$Tools = Join-Path $Root "tools"
$LocalUrl = "http://127.0.0.1:8510"
$Port = 8510

if (-not (Test-Path $Python)) { Fail "environnement Python introuvable : $Python" }

Step "Liberation du port $Port"
Stop-Port $Port

Step "Regeneration de la page a partir du notebook execute"
& $Python -X utf8 (Join-Path $Root "scripts\build_site_v2.py") | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "regeneration de la page impossible (presentation_data.json present ?)" }
Write-Host "Page OK : reports\demo\index.html" -ForegroundColor DarkGreen

Step "Demarrage du serveur de demonstration"
$outLog = Join-Path $LogDir "demo_http.out.log"
$errLog = Join-Path $LogDir "demo_http.err.log"
$demoDir = Join-Path $Root "reports\demo"
Start-Process -FilePath $Python `
    -ArgumentList @("-m", "http.server", "$Port", "--bind", "127.0.0.1", "--directory", "`"$demoDir`"") `
    -WorkingDirectory $Root -WindowStyle Hidden `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog | Out-Null
if (-not (Wait-Http $LocalUrl 60)) {
    Fail "le serveur de demo ne repond pas. Voir logs\demo_http.err.log"
}
Write-Host "Demo locale OK : $LocalUrl" -ForegroundColor DarkGreen

$urlToShow = $LocalUrl

if ($NoTunnel) {
    if (-not $NoBrowser) { try { Start-Process $LocalUrl } catch {} }
    Read-Host "Appuie sur Entree ICI pour tout arreter"
    Step "Arret de la demonstration"
    Stop-Port $Port
    Write-Host "Arrete. Tu peux fermer cette fenetre." -ForegroundColor Yellow
    exit 0
}

Step "Ouverture du navigateur (mode local)"
if (-not $NoBrowser) { try { Start-Process $LocalUrl } catch {} }

Step "Ouverture du tunnel public (cloudflared)"
$Cloudflared = Join-Path $Tools "cloudflared.exe"
if (-not (Test-Path $Cloudflared)) { Fail "cloudflared introuvable : $Cloudflared" }

$cfLog = Join-Path $LogDir "cloudflared_demo.log"
if (Test-Path $cfLog) { Remove-Item $cfLog -Force }
$cfProc = Start-Process -FilePath $Cloudflared `
    -ArgumentList @("tunnel", "--url", "http://127.0.0.1:$Port", "--no-autoupdate") `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogDir "cloudflared_demo.out.log") `
    -RedirectStandardError $cfLog -PassThru

$publicUrl = $null
for ($i = 0; $i -lt 75; $i++) {
    Start-Sleep -Seconds 2
    if (Test-Path $cfLog) {
        $content = Get-Content $cfLog -Raw -ErrorAction SilentlyContinue
        $match = [regex]::Match($content, "https://[a-z0-9-]+\.trycloudflare\.com")
        if ($match.Success) { $publicUrl = $match.Value; break }
    }
}
if (-not $publicUrl) {
    Write-Host "ATTENTION : lien public indisponible (connexion internet ?)" -ForegroundColor Yellow
    Write-Host "La demo tourne quand meme en local : $LocalUrl" -ForegroundColor Yellow
} else {
    Set-Content -LiteralPath (Join-Path $Root "dernier-lien-demo.txt") -Value $publicUrl -Encoding ASCII
    try { Set-Clipboard -Value $publicUrl } catch {}
    $urlToShow = $publicUrl
    Write-Host ""
    Write-Host "  ==============================================" -ForegroundColor Green
    Write-Host "   LIEN A PARTAGER AVEC TES AMIS (copie) :" -ForegroundColor Green
    Write-Host ""
    Write-Host "   $publicUrl" -ForegroundColor White
    Write-Host ""
    Write-Host "  ==============================================" -ForegroundColor Green
    Write-Host "  Valable tant que cette fenetre est ouverte (PC allume)." -ForegroundColor DarkGray
    Write-Host ""
}

Write-Host "  Demo locale : $LocalUrl   |   partagee : $urlToShow" -ForegroundColor DarkGray
Read-Host "Appuie sur Entree ICI pour tout arreter"

Step "Arret de la demonstration"
try { Stop-Process -Id $cfProc.Id -Force -ErrorAction SilentlyContinue } catch {}
Stop-Port $Port
Write-Host "Tout est arrete. Tu peux fermer cette fenetre." -ForegroundColor Yellow
Start-Sleep -Seconds 2
