# Run visualization of the trained PPO model with GUI
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "PPO Agent Traffic Simulation Visualization" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting SUMO GUI with trained PPO agent controlling traffic light..." -ForegroundColor Yellow
Write-Host "Watch the simulation for 3 episodes" -ForegroundColor Yellow
Write-Host ""

# Activate virtual environment if it exists
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
}

# Run the visualization script
python visualize_model.py

Write-Host ""
Write-Host "Visualization complete!" -ForegroundColor Green
