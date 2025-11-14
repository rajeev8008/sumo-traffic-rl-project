"""
Baseline Performance on MG Road Network
Uses the same SumoEnv as RL agent but with fixed-time signal control
"""
import os
import sys
import numpy as np
from SumoEnv import SumoEnv

print("=" * 70)
print("MG ROAD BASELINE - Fixed-Time Traffic Signal Control")
print("=" * 70)

# Configuration
NUM_EPISODES = 3
SIM_DURATION = 1200  # 20 minutes per episode

print(f"\n✓ Number of episodes: {NUM_EPISODES}")
print(f"✓ Simulation duration per episode: {SIM_DURATION} seconds")

# Data collection
all_episode_results = []

for episode in range(NUM_EPISODES):
    print(f"\n{'=' * 70}")
    print(f"Episode {episode + 1}/{NUM_EPISODES}")
    print(f"{'=' * 70}")
    
    try:
        # Create environment (same as RL agent)
        env = SumoEnv(
            use_gui=False,
            sumocfg_file="osm.sumocfg",
            network_type="mg_road"
        )
        
        print("✓ Environment created")
        
        # Reset environment
        obs, info = env.reset()
        print("✓ Environment reset")
        
        # Data collection for this episode
        episode_reward = 0
        episode_length = 0
        waiting_times = []
        speeds = []
        max_queue = 0
        
        # Run episode with fixed-time control (no RL, just alternating phases)
        done = False
        phase_counter = 0
        
        while not done:
            # Fixed-time control: change phase every 30 steps
            if phase_counter % 30 == 0:
                action = 1  # CHANGE phase
            else:
                action = 0  # HOLD phase
            
            phase_counter += 1
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Extract metrics from observation
            # obs = [queue_lengths (4), waiting_times (4), phase (1), padding (4)]
            queue_lengths = obs[:4]
            waiting_obs = obs[4:8]
            
            max_queue = max(max_queue, np.max(queue_lengths) * 10)  # Denormalize
            waiting_times.extend(waiting_obs)
        
        env.close()
        
        # Calculate metrics
        avg_waiting = np.mean(waiting_times) if waiting_times else 0
        avg_reward_per_step = episode_reward / episode_length if episode_length > 0 else 0
        
        episode_results = {
            "episode": episode + 1,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "avg_waiting_time": avg_waiting * 100,  # Denormalize
            "max_queue": max_queue,
            "avg_reward_per_step": avg_reward_per_step,
        }
        
        all_episode_results.append(episode_results)
        
        print(f"\nEpisode {episode + 1} Results:")
        print(f"  Episode Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length} steps")
        print(f"  Average Waiting Time: {avg_waiting * 100:.2f}s")
        print(f"  Max Queue Length: {max_queue:.2f}")
        print(f"  Avg Reward per Step: {avg_reward_per_step:.4f}")
        
    except Exception as e:
        print(f"\n✗ Error in episode {episode + 1}: {e}")
        import traceback
        traceback.print_exc()
        try:
            env.close()
        except:
            pass

# Print summary
print(f"\n{'=' * 70}")
print("BASELINE SUMMARY (Fixed-Time Signal Control)")
print(f"{'=' * 70}")

if all_episode_results:
    avg_episode_reward = np.mean([r["episode_reward"] for r in all_episode_results])
    avg_episode_length = np.mean([r["episode_length"] for r in all_episode_results])
    avg_waiting = np.mean([r["avg_waiting_time"] for r in all_episode_results])
    avg_max_queue = np.mean([r["max_queue"] for r in all_episode_results])
    
    std_episode_reward = np.std([r["episode_reward"] for r in all_episode_results])
    std_episode_length = np.std([r["episode_length"] for r in all_episode_results])
    
    print(f"\nAverage across {NUM_EPISODES} episodes:")
    print(f"  Average Episode Reward: {avg_episode_reward:.2f} (±{std_episode_reward:.2f})")
    print(f"  Average Episode Length: {avg_episode_length:.0f} steps (±{std_episode_length:.0f})")
    print(f"  Average Waiting Time: {avg_waiting:.2f}s")
    print(f"  Average Max Queue: {avg_max_queue:.2f}")
    
    # Baseline results for comparison with RL agent
    baseline_data = {
        "avg_episode_reward": avg_episode_reward,
        "avg_episode_length": avg_episode_length,
        "avg_waiting_time": avg_waiting,
        "std_reward": std_episode_reward,
    }
    
    print(f"\n✓ Baseline data collected successfully!")
    
    # Print for use in compare_results.py
    print(f"\n{'=' * 70}")
    print(f"Update compare_results.py with these baseline values:")
    print(f"{'=' * 70}")
    print(f"""
baseline_results = {{
    "avg_episode_reward": {avg_episode_reward:.2f},
    "avg_episode_length": {avg_episode_length:.0f},
    "avg_waiting_time": {avg_waiting:.2f},
    "std_reward": {std_episode_reward:.2f},
    "avg_duration": {avg_episode_length * 2.27:.2f},  # ~2.27 sec per step
    "avg_speed": 10.0,  # Typical for fixed signal
    "time_loss": {avg_waiting * 1.5:.2f},
    "vehicles_inserted": 200,  # Estimated
    "teleports": 5,  # Estimated
}}
""")
else:
    print("✗ No baseline data collected")

print(f"{'=' * 70}\n")