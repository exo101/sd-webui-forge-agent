param(
    [string]$Python = "..\..\..\system\python\python.exe",
    [string]$OutputDir = "dist\sd-webui-agent-encrypted"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Builder = Join-Path $Root "protect_agent_local.py"
$Out = Join-Path $Root $OutputDir

& $Python $Builder --out $Out
if ($LASTEXITCODE -ne 0) {
    throw "Encrypted Agent build failed."
}

Write-Host "[Agent Protect] Encrypted build ready:"
Write-Host $Out
