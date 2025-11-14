# PPO Agent for SUMO mg_road Traffic Control

This directory contains a **Proximal Policy Optimization (PPO)** agent trained to control traffic signals on the mg_road environment in SUMO. The agent is designed with anti-overfitting measures to ensure good generalization across different traffic scenarios in Bengaluru and beyond.

## Overview

### Project Goals

1. **Train a PPO agent** to optimize traffic signal control on mg_road
2. **Prevent overfitting** through regularization, validation on multiple seeds, and early stopping
3. **Evaluate generalization** to ensure the agent performs well on unseen scenarios
4. **Monitor training progress** with detailed logging and callbacks

### Key Anti-Overfitting Strategies

- **L2 Regularization**: Weight decay built into the optimizer
- **Entropy Coefficient**: Encourages exploration to avoid local optima (ent_coef = 0.02)
- **Gradient Clipping**: Prevents exploding gradients (max_grad_norm = 0.5)
- **Multi-seed Validation**: Evaluates on different random seeds during training
- **Early Stopping**: Stops training if validation performance plateaus
- **Network Design**: Conservative 3-layer architecture [256, 256, 128] with orthogonal initialization
- **Deterministic Evaluation**: Tests on unseen seeds with both deterministic and stochastic policies

## File Structure

```
.
├── ppo_agent.py              # Core PPO agent module (configuration, utilities)
├── train_ppo.py              # Training script with multi-seed validation
├── evaluate_ppo.py           # Comprehensive evaluation script
├── SumoEnv.py                # SUMO environment wrapper (existing)
├── requirements.txt          # Dependencies
├── models/
│   └── ppo_mg_road/
│       ├── best_model.zip    # Best model (saved during training)
│       └── final_model.zip   # Final model at end of training
├── logs/
│   ├── ppo_training/
│   │   ├── config.json       # Training configuration
│   │   └── ...               # TensorBoard logs
│   └── ppo_evaluation/
│       └── evaluation_results.json  # Evaluation metrics
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.10+
- SUMO 1.19.0+ with SUMO_HOME set
- pip

### Setup Steps

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify SUMO installation**:
   ```bash
   # Check if SUMO_HOME is set
   echo $env:SUMO_HOME  # PowerShell
   
   # If not set, add to your environment
   # Windows: Set-Item -Path Env:SUMO_HOME -Value "C:\Program Files\SUMO"
   ```

3. **Verify environment setup**:
   ```bash
   python -c "from SumoEnv import SumoEnv; print('Environment OK')"
   ```

## Usage

### 1. Training

Train a new PPO agent on mg_road:

```bash
# Basic training (default: 500k timesteps)
python train_ppo.py

# Custom training parameters
python train_ppo.py --timesteps 1000000 --eval-freq 20000

# Training without GUI (faster)
python train_ppo.py --no-gui
```

**Training Parameters** (in `train_ppo.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOTAL_TIMESTEPS` | 500,000 | Total training steps |
| `EVAL_FREQ` | 10,000 | Evaluate every N steps |
| `N_EVAL_EPISODES` | 5 | Episodes per evaluation |
| `EARLY_STOPPING_PATIENCE` | 5 | Stop if no improvement for N evals |
| `TRAIN_SEED` | 42 | Training seed for reproducibility |
| `EVAL_SEEDS` | [100, 200, 300] | Seeds for validation |

**Training Output**:
- `models/ppo_mg_road/best_model.zip` - Best model during training
- `models/ppo_mg_road/final_model.zip` - Final model after training
- `logs/ppo_training/` - TensorBoard logs (view with `tensorboard --logdir logs/`)
- `logs/ppo_training/config.json` - Training configuration

### 2. Evaluation

Evaluate a trained model for performance and overfitting detection:

```bash
# Evaluate best model
python evaluate_ppo.py --model models/ppo_mg_road/best_model.zip

# Evaluate with custom seeds
python evaluate_ppo.py --model models/ppo_mg_road/best_model.zip --seeds 100 200 300 400 500

# More episodes per seed
python evaluate_ppo.py --model models/ppo_mg_road/best_model.zip --episodes 5
```

**Evaluation Metrics**:

| Metric | Meaning |
|--------|---------|
| **Training Seed Performance** | Reward on original training seed |
| **Other Seeds Avg** | Average reward on unseen seeds |
| **Performance Drop %** | (Train - Other) / Train × 100 |
| **Generalization Std** | Variance in performance across seeds |
| **Is Overfitted** | If performance drop > 10% |

**Evaluation Output**:
- `logs/ppo_evaluation/evaluation_results.json` - Detailed results
- Console output with overfitting analysis and generalization metrics

### 3. TensorBoard Monitoring

Monitor training in real-time:

```bash
tensorboard --logdir logs/ppo_training
# Then open http://localhost:6006 in browser
```

## PPO Hyperparameters

All hyperparameters are defined in `PPOAgentConfig` in `ppo_agent.py`:

### Network Architecture
```python
NET_ARCH = [256, 256, 128]  # 3-layer neural network
ACTIVATION_FN = "relu"
```

### PPO Algorithm
```python
LEARNING_RATE = 3e-4        # Conservative learning rate
N_STEPS = 2048              # Rollout buffer size
BATCH_SIZE = 64             # Mini-batch size
N_EPOCHS = 10               # Policy updates per batch
GAMMA = 0.99                # Discount factor
GAE_LAMBDA = 0.95           # Generalized Advantage Estimation
```

### Regularization (Anti-Overfitting)
```python
ENT_COEF = 0.02             # Entropy bonus (explore more)
VF_COEF = 0.5               # Value function weight
MAX_GRAD_NORM = 0.5         # Gradient clipping
CLIP_RANGE = 0.2            # PPO clipping range
```

## Environment Details

**Observation Space** (13-dimensional):
- 4 queue lengths (one per incoming lane)
- 1 current phase indicator (0 or 1)
- 4 emergency vehicle flags
- 4 bus vehicle flags

**Action Space**:
- 0: Keep current phase
- 1: Change to next phase

**Reward Function** (multi-objective):
- Penalizes vehicle queues
- Penalizes waiting time
- Bonus for buses (priority)
- Implicit emergency handling through observation

## Expected Performance

### Training Curve
- **Early stages**: High variance, gradual improvement
- **Mid-stage** (100k-300k steps): Steady learning
- **Late stage** (300k+ steps): Performance plateaus or slight overfitting

### Typical Results
- **Training seed reward**: -15 to -25 (negative rewards due to penalties)
- **Other seeds reward**: -18 to -28 (slightly worse, indicating some overfitting)
- **Generalization std**: 2-5 (measure of consistency)
- **Performance drop**: < 10% (good generalization)

## Troubleshooting

### Issue: SUMO not found
```bash
# Set SUMO_HOME
Set-Item -Path Env:SUMO_HOME -Value "C:\Program Files\SUMO"  # Windows
export SUMO_HOME=/usr/share/sumo  # Linux
```

### Issue: Out of memory during training
```python
# In train_ppo.py, reduce:
TrainingConfig.N_STEPS = 1024  # Default 2048
TrainingConfig.BATCH_SIZE = 32  # Default 64
```

### Issue: Training too slow
```bash
# Use --no-gui flag
python train_ppo.py --no-gui

# Reduce eval frequency
python train_ppo.py --eval-freq 20000
```

### Issue: Model overfitting (performance drop > 10%)
Try in `ppo_agent.py`:
```python
ENT_COEF = 0.05             # Increase exploration
LEARNING_RATE = 1e-4        # Reduce learning rate
N_EPOCHS = 5                # Reduce policy updates
```

## Advanced Usage

### Load and Use Trained Model

```python
from ppo_agent import load_ppo_agent
from SumoEnv import SumoEnv

# Load model
agent = load_ppo_agent("models/ppo_mg_road/best_model.zip")

# Use in simulation
env = SumoEnv(use_gui=True, sumocfg_file="SUMO_Trinity_Traffic_sim/osm.sumocfg")
obs, _ = env.reset()

for _ in range(1000):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    if done:
        break

env.close()
```

### Custom Training Loop

```python
from ppo_agent import create_ppo_agent, PPOAgentConfig
from SumoEnv import SumoEnv

# Create environment
env = SumoEnv(use_gui=False, sumocfg_file="SUMO_Trinity_Traffic_sim/osm.sumocfg")

# Create custom config
config = PPOAgentConfig()
config.LEARNING_RATE = 1e-4

# Create agent
agent = create_ppo_agent(env, use_config=config)

# Train
agent.learn(total_timesteps=100000)
agent.save("my_model.zip")
```

## Generalization Strategy

To improve generalization beyond mg_road:

1. **Randomize traffic patterns**: Use different random seeds during training
2. **Vary network parameters**: The environment could expose different network topologies
3. **Multi-task learning**: Train on multiple environments simultaneously (if available)
4. **Domain randomization**: Add noise to observations during training
5. **Curriculum learning**: Start with simple traffic, gradually increase complexity

## References

- Stable-Baselines3 Docs: https://stable-baselines3.readthedocs.io/
- PPO Paper: https://arxiv.org/abs/1707.06347
- SUMO Documentation: https://sumo.dlr.de/
- Gymnasium Documentation: https://gymnasium.farama.org/

## Contributing

To improve this PPO agent:

1. Modify hyperparameters in `PPOAgentConfig`
2. Add new metrics to `evaluate_ppo.py`
3. Implement curriculum learning in `train_ppo.py`
4. Test on different SUMO scenarios

## License

[Your License Here]

## Notes

- All paths in training/evaluation assume PowerShell on Windows (can be adapted to Linux/Mac)
- Training takes ~2-4 hours on CPU depending on hardware
- GPU acceleration recommended for faster training (set `device="cuda"` in `ppo_agent.py`)
- Evaluation is deterministic across runs with fixed seeds
