param(
    [switch]$DryRun,
    [int]$Limit = 0,
    [int]$Workers = 8,
    [int]$Retries = 10,
    [double]$RetrySleep = 2.0,
    [int]$Timeout = 0,
    [int]$MaxTokens = 0,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
$EvalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) { throw "Python was not found." }
if ($Workers -lt 1) { throw "Workers must be at least 1." }

Push-Location $EvalRoot
try {
    $arguments = @("run.py", "--task", "all", "--model-config", "configs/model-qwen-local.json", "--workers", $Workers, "--retries", $Retries, "--retry-sleep", $RetrySleep, "--include-model-in-output")
    if ($DryRun) { $arguments += "--dry-run" }
    if ($Limit -gt 0) { $arguments += @("--limit", $Limit) }
    if ($Timeout -gt 0) { $arguments += @("--timeout", $Timeout) }
    if ($MaxTokens -gt 0) { $arguments += @("--max-tokens", $MaxTokens) }
    if ($Debug) { $arguments += "--debug" }
    & $Python.Source @arguments
    if ($LASTEXITCODE -ne 0) { throw "Qwen run failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}
Write-Host "Qwen: all tasks completed." -ForegroundColor Green
