param(
    [switch]$DryRun,
    [string]$Task = "all",
    [int]$Limit = 0,
    [int]$Workers = 32,
    [int]$Retries = 10,
    [double]$RetrySleep = 2.0,
    [int]$Timeout = 600,
    [int]$MaxTokens = 0,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
$EvalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelConfig = Join-Path $EvalRoot "configs\model-qwen-3.5-4b.json"
$Python = Get-Command python -ErrorAction SilentlyContinue

if (-not $Python) {
    throw "Python was not found."
}
if (-not (Test-Path -LiteralPath $ModelConfig -PathType Leaf)) {
    throw "Model config was not found: $ModelConfig"
}
$ModelProfile = Get-Content -LiteralPath $ModelConfig -Raw | ConvertFrom-Json
$ModelTag = ([string]$ModelProfile.model -replace '[^\p{L}\p{Nd}_-]', '-').Trim('-')
if ($Workers -lt 1) {
    throw "Workers must be at least 1."
}

Push-Location $EvalRoot
try {
    $runStartedAt = Get-Date
    $arguments = @(
        "run.py",
        "--task", $Task,
        "--model-config", $ModelConfig,
        "--workers", $Workers,
        "--retries", $Retries,
        "--retry-sleep", $RetrySleep,
        "--timeout", $Timeout,
        "--include-model-in-output"
    )

    if ($DryRun) {
        $arguments += "--dry-run"
    }
    if ($Limit -gt 0) {
        $arguments += @("--limit", $Limit)
    }
    if ($MaxTokens -gt 0) {
        $arguments += @("--max-tokens", $MaxTokens)
    }
    if ($Debug) {
        $arguments += "--debug"
    }

    Write-Host "Running Qwen3.5-4B task $Task with $Workers workers..." -ForegroundColor Cyan
    & $Python.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Qwen3.5-4B run failed with exit code $LASTEXITCODE."
    }

    if (-not $DryRun) {
        $summaries = Get-ChildItem -LiteralPath (Join-Path $EvalRoot "results") -File -Filter "*-$ModelTag.summary.json" |
            Where-Object { $_.LastWriteTime -ge $runStartedAt.AddSeconds(-1) }
        if (-not $summaries) {
            throw "Qwen run produced no summary files."
        }

        $failedTasks = @(
            foreach ($summaryFile in $summaries) {
                $summary = Get-Content -LiteralPath $summaryFile.FullName -Raw | ConvertFrom-Json
                if ([int]$summary.parse_errors -gt 0) {
                    "{0} ({1} parse errors)" -f $summary.task_code, $summary.parse_errors
                }
            }
        )
        if ($failedTasks.Count -gt 0) {
            throw "Qwen run completed with invalid model responses: $($failedTasks -join ', ')."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "Qwen3.5-4B: all tasks completed." -ForegroundColor Green
