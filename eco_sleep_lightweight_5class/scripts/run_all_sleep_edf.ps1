param(
  [switch]$AllowCpu,
  [switch]$SmokeTrain,
  [switch]$ForceRebuild,
  [string]$Config = "configs\sleep_edf_5class.yaml"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

chcp 65001 > $null
$utf8 = [System.Text.UTF8Encoding]::new()
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$repo = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $repo

if ($env:PYTHONPATH) {
  $env:PYTHONPATH = "$repo\src;$env:PYTHONPATH"
}
else {
  $env:PYTHONPATH = "$repo\src"
}

$python = (Get-Command python -ErrorAction Stop).Source

function Invoke-PythonStep {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [string[]]$Args
  )

  & $python @Args
  if ($LASTEXITCODE -ne 0) {
    Write-Error "$Name failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
  }
}

$preprocessArgs = @("scripts\preprocess_sleep_edf.py", "--config", $Config)
if ($ForceRebuild) {
  $preprocessArgs += "--force_rebuild"
}

$trainArgs = @("scripts\train_sleep_edf.py", "--config", $Config)
if ($AllowCpu) {
  $trainArgs += "--allow_cpu"
}
if ($SmokeTrain) {
  $trainArgs += @("--smoke", "--smoke_epochs", "2", "--smoke_splits", "1")
}

Invoke-PythonStep -Name "compile_check" -Args @("scripts\compile_check.py")
Invoke-PythonStep -Name "preprocess" -Args $preprocessArgs
Invoke-PythonStep -Name "diagnose" -Args @("scripts\diagnose_data.py", "--config", $Config, "--dataset", "sleep_edf")
Invoke-PythonStep -Name "train" -Args $trainArgs
Invoke-PythonStep -Name "eval" -Args @("scripts\eval_sleep_edf.py", "--config", $Config)

$lastRunPath = Join-Path $repo "runs\last_run.txt"
if (-not (Test-Path $lastRunPath)) {
  Write-Error "runs\last_run.txt not found."
  exit 2
}

$runDir = (Get-Content -LiteralPath $lastRunPath -Encoding UTF8 | Select-Object -First 1).Trim()
$summaryPath = Join-Path $runDir "eval\summary_metrics.csv"

Write-Host ("run_dir=" + $runDir)
Write-Host ("summary_metrics=" + $summaryPath)
