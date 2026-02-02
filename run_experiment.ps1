# Run Experiment Batch Script
# 1. Train Model (2016-2024)
# 2. Run Simulation (2025)

Write-Host "=== Starting Experiment ==="

# 1. Train
Write-Host "Step 1: Training Model..."
python scripts/train_experiment_model.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed!"
    exit 1
}

# 2. Simulate
Write-Host "Step 2: Running Simulation..."
python scripts/simulate_strategy_comparison.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Simulation failed!"
    exit 1
}

Write-Host "=== Experiment Completed Successfully ==="
Write-Host "Check 'simulation_report.md' for results."
