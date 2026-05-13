$ErrorActionPreference = "Continue"
Set-Location "D:\Migration\Migrate-main"
$env:FULL_PIPELINE_STAMP = "20260512_111740_cov50_stage7_full"
python -u "run_full_pipeline_cov50.py" > "D:\Migration\Migrate-main\experiments\full_pipeline_20260512_111740_cov50_stage7_full\full_pipeline_20260512_111740_cov50_stage7_full.log" 2>&1
$code = $LASTEXITCODE
if ($null -eq $code) { $code = 0 }
Set-Content -Path "D:\Migration\Migrate-main\experiments\full_pipeline_20260512_111740_cov50_stage7_full\exit_code.txt" -Value $code
exit $code
