# Run the complete evaluation for all vehicle types
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Running PPO Agent Evaluation (All Vehicle Types)" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment if it exists
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
}

# Run the evaluation
python evaluate_all_types.py

Write-Host ""
Write-Host "Evaluation completed!" -ForegroundColor Green
