"""
Training script for PPO agent on SUMO mg_road environment.

Features:
- Multi-core training via SubprocVecEnv for faster iteration.
- Validation on different random seeds during training.
- Anti-overfitting measures (regularization, early stopping)
- Progress monitoring and logging
- Model checkpointing
- CRITICAL FIX: Ensures Observation Space (43 features) is correctly passed to PPO agent.
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Callable, List
import time

# --- New Imports for Parallelization & Stable-Baselines3 ---
from SumoEnv import SumoEnv
from ppo_agent import (
    create_ppo_agent,
    PPOAgentConfig,
    setup_evaluation_callbacks,
)
from stable_baselines3.common.vec_env import SubprocVecEnv 
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
import traci


# --- Global Configuration ---
N_ENVS = 4 

class TrainingConfig:
    """Configuration for training"""
    
    # Training
    TOTAL_TIMESTEPS = 50_000 
    EVAL_FREQ = 10_000  
    N_EVAL_EPISODES = 5  
    EARLY_STOPPING_PATIENCE = 5  
    
    # Environments
    SUMOCFG_FILE = "SUMO_Trinity_Traffic_sim/osm.sumocfg"  
    USE_GUI = False
    
    # Seeds
    TRAIN_SEED = 42
    EVAL_SEEDS = [100, 200, 300]  
    
    # Model saving
    MODEL_DIR = "./models/ppo_mg_road"
    LOG_DIR = "./logs/ppo_training"
    
    # Agent config
    AGENT_CONFIG = PPOAgentConfig()


def setup_directories():
    """Create necessary directories."""
    os.makedirs(TrainingConfig.MODEL_DIR, exist_ok=True)
    os.makedirs(TrainingConfig.LOG_DIR, exist_ok=True)
    print(f"[SETUP] Model directory: {TrainingConfig.MODEL_DIR}")
    print(f"[SETUP] Log directory: {TrainingConfig.LOG_DIR}")


# --- Environment Factory Functions for SubprocVecEnv ---

def create_training_env_fn(seed: int = None):
    """Factory function to create a single training environment instance."""
    env = SumoEnv(
        use_gui=TrainingConfig.USE_GUI,
        sumocfg_file=TrainingConfig.SUMOCFG_FILE,
    )
    if seed is not None:
        env.action_space.seed(seed)
    return env

def make_env_list(num_envs: int, start_seed: int) -> List[Callable]:
    """Returns a list of environment creation functions for SubprocVecEnv."""
    env_fns = []
    for i in range(num_envs):
        env_seed = start_seed + i 
        env_fn = lambda s=env_seed: create_training_env_fn(seed=s) 
        env_fns.append(env_fn)
    return env_fns

def pre_detect_lanes_and_update_env(env: SumoEnv):
    """
    Pre-detect incoming lanes by doing a quick reset and close.
    """
    print("\n[SETUP] Pre-detecting lanes for correct observation space...")
    try:
        obs, _ = env.reset() 
        print(f"[PRE-DETECT] Lanes detected! Observation space: {env.observation_space.shape}")
        env.close()
        print("[PRE-DETECT] Lane detection complete. Observation space locked.")
        return env.observation_space # RETURN THE CORRECT SPACE
    except Exception as e:
        print(f"[PRE-DETECT] Warning: Could not pre-detect lanes: {e}")
        try:
            env.close()
        except Exception:
            pass
        return None


# --- Helper Functions (Restored) ---

def validate_on_seeds(agent, seeds: list, n_episodes: int = 3) -> dict:
    """
    Evaluate agent on different random seeds to detect overfitting.
    (Uses single SumoEnv instances for validation)
    """
    print("\n" + "=" * 70)
    print("[VALIDATION] Evaluating on different random seeds...")
    print("=" * 70)
    
    results = {}
    
    for seed in seeds:
        print(f"\n[VALIDATION] Seed: {seed}")
        
        try:
            traci.close()
        except Exception:
            pass
        
        eval_env = create_training_env_fn(seed=seed)
        
        episode_rewards = []
        
        try:
            for episode in range(n_episodes):
                obs, _ = eval_env.reset()
                done = False
                episode_reward = 0.0
                steps = 0
                
                max_steps = eval_env.max_episode_steps
                
                while not done and steps < max_steps:
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    steps += 1
                
                episode_rewards.append(episode_reward)
                print(f"  Episode {episode + 1}/{n_episodes}: Reward={episode_reward:.3f}, Steps={steps}")
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            results[seed] = {
                "mean": mean_reward,
                "std": std_reward,
                "episodes": episode_rewards,
            }
            
            print(f"  Seed {seed} - Mean: {mean_reward:.3f}, Std: {std_reward:.3f}")
        finally:
            eval_env.close()
            try:
                traci.close()
            except Exception:
                pass
    
    all_means = [results[seed]["mean"] for seed in seeds]
    print(f"\n[VALIDATION] Summary across seeds:")
    print(f"  Overall Mean: {np.mean(all_means):.3f}")
    print(f"  Overall Std: {np.std(all_means):.3f}")
    print(f"  Min: {np.min(all_means):.3f}, Max: {np.max(all_means):.3f}")
    
    return results


def save_training_config(config: TrainingConfig, save_path: str):
    """Save training configuration to JSON."""
    config_dict = {
        "total_timesteps": config.TOTAL_TIMESTEPS,
        "eval_freq": config.EVAL_FREQ,
        "n_eval_episodes": config.N_EVAL_EPISODES,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "sumocfg_file": config.SUMOCFG_FILE,
        "train_seed": config.TRAIN_SEED,
        "eval_seeds": config.EVAL_SEEDS,
        "timestamp": datetime.now().isoformat(),
        "ppo_config": config.AGENT_CONFIG.to_dict(),
    }
    
    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"[SAVE] Training config saved to {save_path}")


# --- Main Training Function ---

def train_ppo(
    total_timesteps: int = TrainingConfig.TOTAL_TIMESTEPS,
    eval_freq: int = TrainingConfig.EVAL_FREQ,
    n_eval_episodes: int = TrainingConfig.N_EVAL_EPISODES,
    early_stopping_patience: int = TrainingConfig.EARLY_STOPPING_PATIENCE,
):
    
    print("\n" + "=" * 70)
    print("[TRAINING] Starting PPO Training for mg_road (PARALLEL MODE)")
    print("=" * 70)
    print(f"[CONFIG] Total timesteps: {total_timesteps:,}")
    print(f"[CONFIG] Eval frequency: {eval_freq:,} steps")
    print(f"[CONFIG] SUMO config: {TrainingConfig.SUMOCFG_FILE}")
    print(f"[CONFIG] Parallel Environments: {N_ENVS}")
    
    setup_directories()
    save_training_config(TrainingConfig, os.path.join(TrainingConfig.LOG_DIR, "config.json"))
    
    # === 1. Pre-detect lanes ===
    temp_env = create_training_env_fn(seed=TrainingConfig.TRAIN_SEED)
    correct_obs_space = pre_detect_lanes_and_update_env(temp_env)
    del temp_env 
    
    if correct_obs_space is None:
        raise RuntimeError("Failed to detect lane count for observation space setup. Cannot proceed.")
    
    # === 2. Create parallel training environment ===
    print(f"\n[SETUP] Creating {N_ENVS} parallel training environments...")
    env_fns = make_env_list(N_ENVS, TrainingConfig.TRAIN_SEED)
    train_env = SubprocVecEnv(env_fns)
    
    # CRITICAL FIX: Set the correct observation space on the SubprocVecEnv 
    # to resolve the matmul (43 vs 13) error during agent initialization.
    train_env.observation_space = correct_obs_space
    
    # === 3. Create parallel evaluation environment ===
    print("[SETUP] Creating parallel evaluation environment...")
    eval_env_fns = make_env_list(1, 42) 
    eval_env = SubprocVecEnv(eval_env_fns)
    
    # CRITICAL FIX: Set the correct observation space for the evaluation environment
    eval_env.observation_space = correct_obs_space

    # === 4. Create agent ===
    print("\n[AGENT] Creating PPO agent...")
    agent = create_ppo_agent(
        env_creator=train_env, # Pass the vectorized training environment
        model_name="ppo_mg_road",
        seed=TrainingConfig.TRAIN_SEED,
        verbose=1,
        use_config=TrainingConfig.AGENT_CONFIG,
    )
    
    # === 5. Setup callbacks ===
    print("[CALLBACKS] Setting up evaluation and monitoring callbacks...")
    eval_callback, overfitting_monitor = setup_evaluation_callbacks(
        eval_env=eval_env, # Pass the vectorized evaluation environment
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=TrainingConfig.MODEL_DIR,
        model_name="best_model",
        patience=early_stopping_patience,
    )
    
    callback_list = CallbackList([eval_callback, overfitting_monitor])
    
    # === 6. Training loop ===
    print("\n[TRAINING] Starting training loop...")
    print("-" * 70)
    
    try:
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=10,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[TRAINING] Training interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise
    
    # Save final model
    final_model_path = os.path.join(TrainingConfig.MODEL_DIR, "final_model.zip")
    agent.save(final_model_path)
    print(f"\n[SAVE] Final model saved to {final_model_path}")
    
    # === 7. Clean up and Final Validation ===
    print("\n[CLEANUP] Closing parallel environments...")
    train_env.close()
    eval_env.close()
    
    time.sleep(1) 
    
    print("\n[VALIDATION] Running validation on multiple seeds...")
    try:
        # Validation uses single, non-vectorized envs to ensure TraCI cleanup
        validation_results = validate_on_seeds(
            agent,
            seeds=TrainingConfig.EVAL_SEEDS,
            n_episodes=3,
        )
        
        # Save validation results
        val_results_path = os.path.join(TrainingConfig.LOG_DIR, "validation_results.json")
        with open(val_results_path, "w") as f:
            json.dump(validation_results, f, indent=2)
        print(f"[SAVE] Validation results saved to {val_results_path}")
    except Exception as e:
        print(f"[WARNING] Validation failed: {e}. Skipping validation.")
        validation_results = {}
    
    print("\n" + "=" * 70)
    print("[TRAINING] Training complete!")
    print(f"[INFO] Best model: {os.path.join(TrainingConfig.MODEL_DIR, 'best_model.zip')}")
    print(f"[INFO] Final model: {final_model_path}")
    print(f"[INFO] Logs: {TrainingConfig.LOG_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for mg_road traffic control")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TrainingConfig.TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TrainingConfig.TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=TrainingConfig.EVAL_FREQ,
        help=f"Evaluation frequency (default: {TrainingConfig.EVAL_FREQ:,})",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run simulation without GUI",
    )
    
    args = parser.parse_args()
    
    # Update config based on CLI args
    TrainingConfig.TOTAL_TIMESTEPS = args.timesteps
    TrainingConfig.EVAL_FREQ = args.eval_freq
    TrainingConfig.USE_GUI = not args.no_gui
    
    # Start training
    train_ppo(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
    )