import os
import sys
import numpy as np
from stable_baselines3 import PPO
from SumoEnv import SumoEnv

print("=" * 60)
print("Evaluating PPO Agent on MG Road Network")
print("=" * 60)

# Configuration
MODEL_PATH = "models/ppo_mg_road/ppo_mg_road_final_model.zip"
NUM_EPISODES = 1  # Single episode for GUI visualization

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] Model not found at {MODEL_PATH}")
    print("Please train the model first using train_mg_road.py")
    exit(1)

print(f"\nLoading model from {MODEL_PATH}...")
try:
    model = PPO.load(MODEL_PATH)
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# Create evaluation environment
print("\nCreating evaluation environment for MG Road...")
try:
    env = SumoEnv(
        use_gui=True,  # Enable GUI to watch the simulation
        sumocfg_file="osm_fast.sumocfg",  # Use fast config matching training
        network_type="mg_road"
    )
    print("[OK] Environment created successfully")
except Exception as e:
    print(f"[ERROR] Failed to create environment: {e}")
    exit(1)

# Run evaluation
print(f"\n{'=' * 60}")
print(f"Running {NUM_EPISODES} evaluation episodes...")
print(f"{'=' * 60}\n")

episode_rewards = []
episode_lengths = []

for episode in range(NUM_EPISODES):
    observation, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False
    
    print(f"Episode {episode + 1}/{NUM_EPISODES}:")
    
    while not done:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
    
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    
    print(f"  [OK] Episode Reward: {episode_reward:.2f}")
    print(f"  [OK] Episode Length: {episode_length} steps\n")

# Print summary
print(f"{'=' * 60}")
print("Evaluation Summary")
print(f"{'=' * 60}")
print(f"Average Episode Reward: {np.mean(episode_rewards):.2f} (±{np.std(episode_rewards):.2f})")
print(f"Average Episode Length: {np.mean(episode_lengths):.0f} (±{np.std(episode_lengths):.0f})")
print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
print(f"Worst Episode Reward: {np.min(episode_rewards):.2f}")
print(f"{'=' * 60}")

env.close()
print("\n[OK] Evaluation complete!")
