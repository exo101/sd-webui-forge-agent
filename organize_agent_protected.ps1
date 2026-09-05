$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Agent protected build organizer"

try {
    $Root = "D:\AI\sd-webui-forge-neo-v3.6.5"
    $DevSource = Join-Path $Root "webui\extensions\sd-webui-agent-dev"
    $Plugin = Join-Path $Root "webui\extensions\sd-webui-agent"
    $BackupRoot = "D:\AI\backup"
    $Dev = Join-Path $BackupRoot "sd-webui-agent-dev"
    $Python = Join-Path $Root "system\python\python.exe"
    $Builder = Join-Path $DevSource "protect_agent_local.py"
    $BuildRoot = Join-Path $Root "_agent_encrypted_build"
    $Log = Join-Path $Root "agent_organize.log"

    Start-Transcript -Path $Log -Force | Out-Null
    Write-Host ""
    Write-Host "Agent protected build organizer"
    Write-Host "Development source: $DevSource"
    Write-Host "Developer backup: $Dev"
    Write-Host "Runtime copy:   $Plugin"
    Write-Host ""

    if (!(Test-Path $DevSource)) { throw "Development folder not found: $DevSource" }
    if (!(Test-Path (Join-Path $DevSource "scripts\agent.py"))) { throw "Development agent entry not found" }
    if (!(Test-Path $Python)) { throw "Forge Python not found: $Python" }
    if (!(Test-Path $Builder)) { throw "Builder not found: $Builder" }

    New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
    if (Test-Path $Dev) {
        $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        Rename-Item -LiteralPath $Dev -NewName "sd-webui-agent-dev-old-$Stamp"
    }

    Write-Host "[1/4] Backing up the development copy..."
    Copy-Item -LiteralPath $DevSource -Destination $Dev -Recurse -Force
    if (!(Test-Path (Join-Path $Dev "scripts\agent_tools.py"))) { throw "Development backup verification failed" }

    Write-Host "[2/4] Building the encrypted copy..."
    if (Test-Path $BuildRoot) { Remove-Item -LiteralPath $BuildRoot -Recurse -Force }
    & $Python $Builder --out $BuildRoot
    if ($LASTEXITCODE -ne 0) { throw "Encrypted build failed" }
    if (!(Test-Path (Join-Path $BuildRoot "scripts\agent.agentpkg"))) { throw "Encrypted build verification failed" }

    Write-Host "[3/4] Installing the encrypted copy..."
    $Stage = Join-Path $Root "_agent_protected_stage"
    if (Test-Path $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
    Copy-Item -LiteralPath $BuildRoot -Destination $Stage -Recurse -Force
    if (Test-Path $Plugin) {
        Get-ChildItem -LiteralPath $Plugin -Force | Remove-Item -Recurse -Force
    } else {
        New-Item -ItemType Directory -Force -Path $Plugin | Out-Null
    }
    Get-ChildItem -LiteralPath $Stage -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $Plugin -Recurse -Force
    }
    Remove-Item -LiteralPath $Stage -Recurse -Force
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force

    Write-Host "[4/4] Verifying the installed copy..."
    if (!(Test-Path (Join-Path $Plugin "scripts\agent.agentpkg"))) { throw "Installed package not found" }
    $Head = Get-Content -LiteralPath (Join-Path $Plugin "scripts\agent.py") -TotalCount 2
    if (($Head -join "`n") -notmatch "Protected loader") { throw "Installed agent entry is not a protected loader" }

    Write-Host ""
    Write-Host "Done. Restart WebUI."
    Write-Host "Development source: $DevSource"
    Write-Host "Developer backup: $Dev"
    Write-Host "Runtime copy:     $Plugin"
}
catch {
    Write-Host ""
    Write-Host "FAILED:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Log: $Log"
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
    Write-Host ""
    Read-Host "Press Enter to close"
}
