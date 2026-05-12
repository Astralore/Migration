$ErrorActionPreference = "Continue"
Set-Location "D:\Migration\Migrate-main"
python -u "run_full_pipeline_cov50.py" > "D:\Migration\Migrate-main\experiments\full_pipeline_20260511_1711_cov50_predictor_v2_full\full_pipeline_20260511_1711_cov50_predictor_v2_full.log" 2>&1
$code = $LASTEXITCODE
if ($null -eq $code) { $code = 0 }
Set-Content -Path "D:\Migration\Migrate-main\experiments\full_pipeline_20260511_1711_cov50_predictor_v2_full\exit_code.txt" -Value $code
exit $code
