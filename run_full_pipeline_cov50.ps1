$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Root "experiments\full_pipeline_20260511_1711_cov50_predictor_v2_full"
$LogPath = Join-Path $LogDir "full_pipeline_20260511_1711_cov50_predictor_v2_full.log"
$ExitCodePath = Join-Path $LogDir "exit_code.txt"
$RunnerPath = Join-Path $LogDir "run_detached.ps1"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Remove-Item -Force -ErrorAction SilentlyContinue $ExitCodePath

$runner = @"
`$ErrorActionPreference = "Continue"
Set-Location "$Root"
python -u "run_full_pipeline_cov50.py" > "$LogPath" 2>&1
`$code = `$LASTEXITCODE
if (`$null -eq `$code) { `$code = 0 }
Set-Content -Path "$ExitCodePath" -Value `$code
exit `$code
"@

Set-Content -Path $RunnerPath -Value $runner -Encoding UTF8

$proc = Start-Process -FilePath "powershell" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunnerPath) `
    -WorkingDirectory $Root `
    -PassThru

Write-Output $proc.Id
