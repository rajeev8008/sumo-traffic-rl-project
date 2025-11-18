"""
PPO Agent Training Script - AMBULANCE PRIORITY VERSION (V3)

Trains a PPO (Proximal Policy Optimization) agent using Stable-Baselines3
to optimize traffic light control on MG Road network with AMBULANCE PRIORITIZATION.

IMPROVEMENTS FROM V2:
- Ambulance-focused reward function (5.0 bonus per ambulance!)
- Heavy penalty for ambulance waiting (-0.1 * avg_ambulance_wait)
- Gentler penalties for other vehicles (softer congestion management)
- Same stable hyperparameters as v2

Features:
- PPO algorithm from Stable-Baselines3
- Callbacks for monitoring training progress
- Model checkpointing every 10K timesteps
- Tensorboard logging for visualization
- CUDA GPU acceleration (automatic detection)
- Automatic environment reset after episodes

Enhanced Training Configuration:
- Total timesteps: 200,000 (same as v2 for fair comparison)
- Learning rate: 5e-4 (same as v2)
- Batch size: 128 (same as v2)
- Number of epochs per update: 20 (same as v2)
- Gamma (discount): 0.99 (high future value)
- GAE lambda: 0.95 (balanced advantage estimation)

REWARD FUNCTION (V3 - AMBULANCE PRIORITY - FIXED):
- ambulance_wait_penalty: -1.0 * avg_ambulance_wait (10x STRONGER! CRITICAL!)
- flow_bonus: 0.5 * (total/100) (10x STRONGER - primary objective)
- queue_penalty: -0.01 * avg_queue (10x STRONGER)
- wait_penalty: -0.005 * avg_wait (10x STRONGER)
- congestion_penalty: -0.05 * (queue-30) if queue>30 (5x STRONGER)
- emergency_bonus: 0.2 per vehicle (minimal)
- bus_bonus: 0.05 per vehicle (minimal)
- ambulance_bonus: REMOVED (was causing gaming behavior)
"""

import sys
import os
import json
from datetime import datetime
import numpy as np

sys.path.insert(0, '.')

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Custom environment
from sumo_mg_road_env import MGRoadEnv


def create_environment():
    """
    Create and return the SUMO environment for training
    
    Returns:
        MGRoadEnv: Configured environment
    """
    env = MGRoadEnv(use_gui=False, test_routes=True)
    return env


def train_ppo(
    total_timesteps=200000,
    learning_rate=5e-4,
    batch_size=128,
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    checkpoint_interval=10000,
    use_tensorboard=True,
    model_save_path="models/ppo_model_v3"
):
    """
    Train PPO agent on traffic light control task with improvements
    
    Args:
        total_timesteps (int): Total environment interactions
        learning_rate (float): PPO learning rate
        batch_size (int): Batch size for training
        n_epochs (int): Number of epochs per update
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        checkpoint_interval (int): Save checkpoint every N timesteps
        use_tensorboard (bool): Enable tensorboard logging
        model_save_path (str): Directory to save models
    
    Returns:
        PPO: Trained PPO model
    """
    
    # Create directories
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("\n" + "="*70)
    print("PPO AGENT TRAINING - AMBULANCE PRIORITY VERSION (V3)")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs per update: {n_epochs}")
    print(f"  - Discount (gamma): {gamma}")
    print(f"  - GAE lambda: {gae_lambda}")
    print(f"  - Checkpoint interval: {checkpoint_interval:,} steps")
    print(f"  - Use tensorboard: {use_tensorboard}")
    print(f"  - Model save path: {model_save_path}")
    print("="*70)
    print("\nREWARD FUNCTION (V3 - AMBULANCE PRIORITY - FIXED):")
    print("  ✓ ambulance_wait_penalty: -1.0 * avg_ambulance_wait (10x STRONGER!)")
    print("  ✓ flow_bonus: 0.5 * (total/100) (primary objective, 10x stronger!)")
    print("  ✓ queue_penalty: -0.01 * avg_queue (10x stronger)")
    print("  ✓ wait_penalty: -0.005 * avg_wait (10x stronger)")
    print("  ✓ congestion_penalty: -0.05 * (queue-30) (5x stronger)")
    print("  ✓ emergency_bonus: 0.2 per vehicle (minimal)")
    print("  ✓ queue_penalty: -0.001 * avg_queue (soft)")
    print("  ✓ wait_penalty: -0.0005 * avg_wait (soft)")
    print("  ✓ flow_bonus: 0.05 * (total/100)")
    print("  ✓ congestion_penalty: -0.01 * (queue-30) if queue>30")
    print("="*70)
    
    # Create environment
    print("\n[1/3] Creating environment...")
    env = create_environment()
    print(f"[OK] Environment created")
    print(f"      - Action space: {env.action_space}")
    print(f"      - Observation space: {env.observation_space}")
    
    # Define checkpoint callback - saves model every checkpoint_interval steps
    print("\n[2/3] Setting up checkpoint callback...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=model_save_path,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False
    )
    print(f"[OK] Checkpoint callback configured (saves every {checkpoint_interval:,} steps)")
    
    # Create PPO model with improvements
    print("\n[3/3] Creating PPO model...")
    
    tensorboard_log = "logs/ppo_logs_v2" if use_tensorboard else None
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=batch_size,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=tensorboard_log,
        verbose=1
    )
    
    print(f"[OK] PPO model created")
    print(f"      - Policy: MlpPolicy")
    print(f"      - Network architecture: 2 hidden layers of 64 units")
    
    # Train model
    print(f"\n[TRAINING] Starting training for {total_timesteps:,} timesteps...")
    print(f"[TRAINING] Expected duration: ~150-180 minutes")
    print()
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            log_interval=10,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user")
        print("[SAVING] Saving current model...")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    print(f"\n[OK] Training completed in {training_duration:.1f} seconds ({training_duration/60:.1f} minutes)")
    
    # Save final model
    print("\n[SAVING] Saving final model...")
    final_model_path = os.path.join(model_save_path, "ppo_final_v3.zip")
    model.save(final_model_path)
    print(f"[OK] Final model saved to {final_model_path}")
    
    # Close environment
    env.close()
    
    return model, training_duration


def save_training_config(config, filename="training_config_v2.json"):
    """
    Save training configuration to JSON file
    
    Args:
        config (dict): Training configuration
        filename (str): Output filename
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[OK] Training config saved to {filename}")


def print_training_summary(duration):
    """
    Print summary of training
    
    Args:
        duration (float): Training duration in seconds
    """
    print("\n" + "="*70)
    print("TRAINING SUMMARY - AMBULANCE PRIORITY VERSION (V3)")
    print("="*70)
    print(f"Training Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print()
    print("Models saved:")
    print("  - Checkpoints: models/ppo_model_v3/ppo_checkpoint_*.zip")
    print("  - Final model: models/ppo_model_v3/ppo_final_v3.zip")
    print()
    print("Training logs:")
    print("  - Tensorboard: logs/ppo_logs_v2/")
    print("    View with: tensorboard --logdir logs/ppo_logs_v2")
    print()
    print("Reward function (V3 - Ambulance Priority):")
    print("  ✓ ambulance_wait_penalty: -1.0 * avg_ambulance_wait (10x STRONGER!)")
    print("  ✓ flow_bonus: 0.5 * (total/100) (primary objective, 10x stronger!)")
    print("  ✓ queue_penalty: -0.01 * avg_queue (10x stronger)")
    print("  ✓ wait_penalty: -0.005 * avg_wait (10x stronger)")
    print("  ✓ congestion_penalty: -0.05 * (queue-30) (5x stronger)")
    print("  ✓ Soft penalties for other vehicles")
    print("  ✓ Flow bonus and congestion penalty")
    print()
    print("Expected improvements over baseline:")
    print("  - Ambulance wait time: 61.7s (baseline) → <30s (target)")
    print("  - Overall reward: Should exceed v2's +759")
    print("  - Emergency vehicle prioritization: Very high!")
    print()
    print("Next steps:")
    print("  1. Evaluate trained model: python evaluate_v2.py")
    print("  2. Compare ambulance metrics with baseline")
    print("  3. Verify <30s wait time for ambulances")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Training configuration
    config = {
        "version": "v3_ambulance_priority",
        "total_timesteps": 200000,
        "learning_rate": 5e-4,
        "batch_size": 128,
        "n_epochs": 20,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "checkpoint_interval": 10000,
        "use_tensorboard": True,
        "improvements": {
            "reward_function": "Ambulance-focused: 5.0 ambulance bonus, -0.1 ambulance wait penalty, soft penalties for others",
            "ambulance_bonus": "5.0 per ambulance (vs 1.0 for emergency in v2)",
            "ambulance_wait_penalty": "-0.1 * avg_ambulance_wait (CRITICAL!)",
            "target": "Ambulance wait time <30s (from 61.7s baseline)",
            "compared_to_v2": "Same hyperparameters, different reward function",
            "training_length": "200K timesteps (same as v2)"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Train PPO agent
    model, duration = train_ppo(
        total_timesteps=config["total_timesteps"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        checkpoint_interval=config["checkpoint_interval"],
        use_tensorboard=config["use_tensorboard"]
    )
    
    # Save configuration
    save_training_config(config)
    
    # Print summary
    print_training_summary(duration)

