# Quick Start Guide: PPO Agent Training & Evaluation

## 5-Minute Setup

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Verify Setup
```powershell
# Check SUMO is accessible
sumo --version

# Check Python imports work
python -c "from SumoEnv import SumoEnv; from ppo_agent import create_ppo_agent; print('✓ Ready to train!')"
```

### 3. Start Training
```powershell
# Fastest option (headless, ~2-4 hours)
python train_ppo.py --no-gui

# Or with custom parameters
python train_ppo.py --no-gui --timesteps 500000 --eval-freq 10000
```

**What to expect**:
- Progress bar showing training progress
- Evaluation every 10k steps
- Model saves to `models/ppo_mg_road/`
- Logs go to `logs/ppo_training/`

## Full Workflow

### Step 1: Train
```powershell
python train_ppo.py --no-gui
```
**Outputs**:
- `models/ppo_mg_road/best_model.zip` ← Use this
- `models/ppo_mg_road/final_model.zip`
- `logs/ppo_training/config.json`

### Step 2: Evaluate
```powershell
python evaluate_ppo.py --model models/ppo_mg_road/best_model.zip
```
**Outputs**:
- Console printout with metrics
- `logs/ppo_evaluation/evaluation_results.json`

### Step 3: Check Results
```powershell
# View JSON results
type logs/ppo_evaluation/evaluation_results.json

# View TensorBoard logs
tensorboard --logdir logs/ppo_training
# Open browser to http://localhost:6006
```

## Key Metrics to Look For

### During Training
- **Ep Rew Mean** (in TensorBoard): Should increase over time (become less negative)
- **Early stopping**: Training stops if validation plateaus for 5 evaluations

### After Evaluation
- **Overfitting Status**: Should be "NO ✓"
- **Performance Drop**: Should be < 10%
- **Generalization Std**: Lower is better (more consistent)

## If Something Goes Wrong

### Training freezes / hangs
```powershell
# Use --no-gui to speed up
python train_ppo.py --no-gui

# Or use smaller batch size (edit train_ppo.py):
# TrainingConfig.TOTAL_TIMESTEPS = 100_000  # Start small
```

### SUMO not found
```powershell
# Set SUMO_HOME (Windows)
$env:SUMO_HOME = "C:\Program Files\SUMO"

# Or permanent (PowerShell):
[System.Environment]::SetEnvironmentVariable("SUMO_HOME", "C:\Program Files\SUMO", "User")
```

### Out of memory
Edit `train_ppo.py`:
```python
TrainingConfig.TOTAL_TIMESTEPS = 100_000  # Smaller
TrainingConfig.N_STEPS = 1024  # Smaller batches
```

## Files Created

| File | Purpose |
|------|---------|
| `ppo_agent.py` | Core PPO implementation + config |
| `train_ppo.py` | Training with multi-seed validation |
| `evaluate_ppo.py` | Evaluation & overfitting detection |
| `PPO_README.md` | Full documentation |
| `QUICKSTART.md` | This file |

## Next Steps

1. **Experiment with hyperparameters** in `ppo_agent.py`:
   - `LEARNING_RATE`: Try 1e-4 to 1e-3
   - `ENT_COEF`: Try 0.01 to 0.1 (higher = more exploration)
   - `NET_ARCH`: Try [512, 512] for larger network

2. **Test generalization** by adding more evaluation seeds in `evaluate_ppo.py`:
   ```python
   EvaluationConfig.EVAL_SEEDS = [100, 200, 300, 400, 500, 600, 700]
   ```

3. **Deploy the model** using the trained agent in your own simulation

## Support

See `PPO_README.md` for detailed documentation and troubleshooting.
