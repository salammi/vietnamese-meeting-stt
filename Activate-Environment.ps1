# Run from PowerShell with:
#   . .\Activate-Environment.ps1
#
# The leading dot is required so the virtual environment stays active
# in the current PowerShell session.

$projectRoot = $PSScriptRoot
$activateScript = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path -LiteralPath $activateScript)) {
    throw "Virtual environment was not found at '$activateScript'. Create it first with: py -3.12 -m venv .venv"
}

Set-Location -LiteralPath $projectRoot
. $activateScript

Write-Host "Activated virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
