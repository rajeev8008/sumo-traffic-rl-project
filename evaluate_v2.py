"""
Evaluation Script - Compare Baseline vs PPO Models

Evaluates trained PPO models and compares performance with baseline (fixed-time) controller.
Shows improvements in queue length, wait time, and per-vehicle-type metrics.

This script can evaluate any PPO model version (v2, v3, etc.) by changing the model_path.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, '.')

from sumo_mg_road_env import MGRoadEnv
from stable_baselines3 import PPO


def load_baseline_metrics(filename="baseline_metrics.json"):
    """Load baseline metrics from JSON"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    if not os.path.exists(filepath):
        print(f"[ERROR] {filename} not found!")
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def get_vehicle_type_stats(env, vehicle_ids):
    """
    Get statistics for each vehicle type
    
    Args:
        env: SUMO environment
        vehicle_ids: List of vehicle IDs
    
    Returns:
        dict: Stats per vehicle type
    """
    import traci
    
    vehicle_types = {
        "motorcycle": [],
        "car": [],
        "auto": [],
        "bus": [],
        "truck": [],
        "ambulance": []
    }
    
    for vid in vehicle_ids:
        try:
            vtype = traci.vehicle.getTypeID(vid)
            wait_time = traci.vehicle.getAccumulatedWaitingTime(vid)
            
            # Classify vehicle
            if vtype.startswith('motorcycle'):
                vehicle_types['motorcycle'].append(wait_time)
            elif vtype.startswith('car'):
                vehicle_types['car'].append(wait_time)
            elif vtype.startswith('auto'):
                vehicle_types['auto'].append(wait_time)
            elif vtype.startswith(('bus', 'coach')):
                vehicle_types['bus'].append(wait_time)
            elif vtype.startswith('truck'):
                vehicle_types['truck'].append(wait_time)
            elif vtype.startswith('ambulance'):
                vehicle_types['ambulance'].append(wait_time)
        except:
            pass
    
    stats = {}
    for vtype, wait_times in vehicle_types.items():
        if wait_times:
            stats[vtype] = {
                "count": len(wait_times),
                "avg_wait_time": float(np.mean(wait_times)),
                "max_wait_time": float(np.max(wait_times)),
                "total_wait_time": float(np.sum(wait_times))
            }
        else:
            stats[vtype] = {
                "count": 0,
                "avg_wait_time": 0,
                "max_wait_time": 0,
                "total_wait_time": 0
            }
    
    return stats


def evaluate_ppo_model(model_path, num_episodes=3, max_steps=3600):
    """
    Evaluate a trained PPO model with full episode length
    
    Args:
        model_path (str): Path to trained model
        num_episodes (int): Number of episodes to evaluate
        max_steps (int): Max steps per episode (3600 = 1 hour, same as training)
    
    Returns:
        dict: Evaluation metrics with per-vehicle-type breakdown
    """
    import traci
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None
    
    # Load environment and model
    env = MGRoadEnv(use_gui=False, test_routes=True)
    model = PPO.load(model_path)
    
    print(f"\n[LOADING] Loaded model from: {model_path}")
    print(f"[CONFIG] Episode length: {max_steps} steps (full hour)")
    
    all_episode_metrics = []
    
    for episode in range(num_episodes):
        print(f"\n[EPISODE {episode + 1}/{num_episodes}]")
        
        obs, info = env.reset()
        
        episode_reward = 0
        episode_steps = 0
        queue_lengths = []
        wait_times = []
        vehicle_counts = []
        all_vehicle_ids = []
        
        for step in range(max_steps):
            # Use trained model to predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Collect metrics
            episode_reward += reward
            episode_steps += 1
            queue_lengths.append(obs[0])  # TL1 queue length
            wait_times.append(obs[1])     # TL1 wait time
            vehicle_counts.append(info['total_vehicles'])
            
            # Collect all vehicle IDs in this step
            try:
                current_vehicles = traci.vehicle.getIDList()
                all_vehicle_ids.extend(current_vehicles)
            except:
                pass
            
            if (step + 1) % 360 == 0:  # Every 100 sim seconds
                print(f"  Step {step + 1}/{max_steps}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Vehicles={info['total_vehicles']}, "
                      f"Avg Queue={np.mean(queue_lengths[-10:]):.1f}")
            
            if terminated or truncated:
                break
        
        # Get per-vehicle-type statistics
        vehicle_type_stats = get_vehicle_type_stats(env, list(set(all_vehicle_ids)))
        
        # Compute episode statistics
        episode_metrics = {
            "episode": episode + 1,
            "steps": episode_steps,
            "total_reward": float(episode_reward),
            "avg_reward_per_step": float(episode_reward / episode_steps) if episode_steps > 0 else 0,
            "avg_queue_length": float(np.mean(queue_lengths)),
            "max_queue_length": float(np.max(queue_lengths)) if queue_lengths else 0,
            "avg_wait_time": float(np.mean(wait_times)),
            "max_wait_time": float(np.max(wait_times)) if wait_times else 0,
            "avg_vehicles": float(np.mean(vehicle_counts)),
            "max_vehicles": int(np.max(vehicle_counts)) if vehicle_counts else 0,
            "vehicle_type_stats": vehicle_type_stats
        }
        
        all_episode_metrics.append(episode_metrics)
        
        print(f"  Episode Reward: {episode_metrics['total_reward']:.2f}")
        print(f"  Avg Queue Length: {episode_metrics['avg_queue_length']:.1f}")
        print(f"  Avg Wait Time: {episode_metrics['avg_wait_time']:.1f} sec")
        print(f"  Avg Vehicles: {episode_metrics['avg_vehicles']:.0f}")
        print(f"  \n  Per-Vehicle-Type Metrics:")
        for vtype in sorted(vehicle_type_stats.keys()):
            stats = vehicle_type_stats[vtype]
            if stats['count'] > 0:
                print(f"    {vtype.upper()}:")
                print(f"      - Avg Wait: {stats['avg_wait_time']:.1f}s | "
                      f"Max Wait: {stats['max_wait_time']:.1f}s | "
                      f"Count: {stats['count']}")
    
    env.close()
    
    # Compute aggregate statistics
    eval_metrics = {
        "model_path": model_path,
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps,
        "timestamp": datetime.now().isoformat(),
        "episodes": all_episode_metrics,
        "aggregate": {
            "avg_reward": float(np.mean([m['total_reward'] for m in all_episode_metrics])),
            "std_reward": float(np.std([m['total_reward'] for m in all_episode_metrics])),
            "avg_queue_length": float(np.mean([m['avg_queue_length'] for m in all_episode_metrics])),
            "std_queue_length": float(np.std([m['avg_queue_length'] for m in all_episode_metrics])),
            "avg_wait_time": float(np.mean([m['avg_wait_time'] for m in all_episode_metrics])),
            "std_wait_time": float(np.std([m['avg_wait_time'] for m in all_episode_metrics])),
            "avg_vehicles": float(np.mean([m['avg_vehicles'] for m in all_episode_metrics])),
            "max_vehicles": int(np.max([m['max_vehicles'] for m in all_episode_metrics])),
        }
    }
    
    # Aggregate vehicle type stats
    vehicle_type_aggregate = {}
    for vtype in ["motorcycle", "car", "auto", "bus", "truck", "ambulance"]:
        counts = []
        wait_times_list = []
        max_waits = []
        for ep in all_episode_metrics:
            if vtype in ep['vehicle_type_stats']:
                stats = ep['vehicle_type_stats'][vtype]
                counts.append(stats['count'])
                if stats['count'] > 0:
                    wait_times_list.append(stats['avg_wait_time'])
                    max_waits.append(stats['max_wait_time'])
        
        if counts and np.sum(counts) > 0:
            vehicle_type_aggregate[vtype] = {
                "avg_count": float(np.mean(counts)),
                "avg_wait_time": float(np.mean(wait_times_list)) if wait_times_list else 0,
                "max_wait_time": float(np.max(max_waits)) if max_waits else 0
            }
    
    eval_metrics["vehicle_type_aggregate"] = vehicle_type_aggregate
    
    return eval_metrics


def calculate_improvement(baseline, ppo_eval):
    """
    Calculate improvement of PPO over baseline
    
    Args:
        baseline (dict): Baseline metrics
        ppo_eval (dict): PPO evaluation metrics
    
    Returns:
        dict: Improvement percentages
    """
    
    baseline_agg = baseline['aggregate']
    ppo_agg = ppo_eval['aggregate']
    
    improvements = {
        "reward": {
            "baseline": baseline_agg['avg_reward'],
            "ppo": ppo_agg['avg_reward'],
            "improvement_pct": ((ppo_agg['avg_reward'] - baseline_agg['avg_reward']) / abs(baseline_agg['avg_reward'])) * 100
        },
        "queue_length": {
            "baseline": baseline_agg['avg_queue_length'],
            "ppo": ppo_agg['avg_queue_length'],
            "improvement_pct": ((baseline_agg['avg_queue_length'] - ppo_agg['avg_queue_length']) / baseline_agg['avg_queue_length']) * 100
        },
        "wait_time": {
            "baseline": baseline_agg['avg_wait_time'],
            "ppo": ppo_agg['avg_wait_time'],
            "improvement_pct": ((baseline_agg['avg_wait_time'] - ppo_agg['avg_wait_time']) / baseline_agg['avg_wait_time']) * 100
        }
    }
    
    return improvements


def print_comparison(baseline, ppo_eval, improvements):
    """Print detailed comparison table with per-vehicle-type breakdown"""
    
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON - BASELINE vs PPO")
    print("="*100)
    
    print(f"\n{'Metric':<35} {'Baseline':<20} {'PPO':<20} {'Improvement':<15}")
    print("-" * 100)
    
    # Reward
    print(f"{'Total Reward':<35} {improvements['reward']['baseline']:<20.2f} "
          f"{improvements['reward']['ppo']:<20.2f} "
          f"{improvements['reward']['improvement_pct']:>+.1f}%")
    
    # Queue length
    print(f"{'Queue Length (vehicles)':<35} {improvements['queue_length']['baseline']:<20.1f} "
          f"{improvements['queue_length']['ppo']:<20.1f} "
          f"{improvements['queue_length']['improvement_pct']:>+.1f}%")
    
    # Wait time
    print(f"{'Wait Time (seconds)':<35} {improvements['wait_time']['baseline']:<20.1f} "
          f"{improvements['wait_time']['ppo']:<20.1f} "
          f"{improvements['wait_time']['improvement_pct']:>+.1f}%")
    
    # Max vehicles
    print(f"{'Max Vehicles in System':<35} {baseline['aggregate']['max_vehicles']:<20} "
          f"{ppo_eval['aggregate']['max_vehicles']:<20} -")
    
    print("=" * 100)
    
    # Per-vehicle-type comparison
    print("\n" + "="*100)
    print("PER-VEHICLE-TYPE COMPARISON")
    print("="*100)
    
    if 'vehicle_type_aggregate' in ppo_eval and ppo_eval['vehicle_type_aggregate']:
        print(f"\n{'Vehicle Type':<20} {'Avg Wait (PPO)':<20} {'Max Wait (PPO)':<20} {'Count':<15}")
        print("-" * 100)
        
        for vtype in sorted(ppo_eval['vehicle_type_aggregate'].keys()):
            stats = ppo_eval['vehicle_type_aggregate'][vtype]
            print(f"{vtype.upper():<20} {stats['avg_wait_time']:<20.1f}s "
                  f"{stats['max_wait_time']:<20.1f}s "
                  f"{stats['avg_count']:<15.0f}")
        
        print("=" * 100)
    
    # Summary
    print("\n📊 OVERALL SUMMARY:")
    print("-" * 100)
    
    if improvements['queue_length']['improvement_pct'] > 0:
        print(f"✓ Queue length IMPROVED by {improvements['queue_length']['improvement_pct']:.1f}%")
    else:
        print(f"✗ Queue length WORSENED by {abs(improvements['queue_length']['improvement_pct']):.1f}%")
    
    if improvements['wait_time']['improvement_pct'] > 0:
        print(f"✓ Wait time IMPROVED by {improvements['wait_time']['improvement_pct']:.1f}%")
    else:
        print(f"✗ Wait time WORSENED by {abs(improvements['wait_time']['improvement_pct']):.1f}%")
    
    if improvements['reward']['improvement_pct'] > 0:
        print(f"✓ Total reward IMPROVED by {improvements['reward']['improvement_pct']:.1f}%")
    else:
        print(f"✗ Total reward WORSENED by {abs(improvements['reward']['improvement_pct']):.1f}%")
    
    print("-" * 100)
    
    # Success criteria
    print("\n🎯 SUCCESS CRITERIA:")
    queue_ok = improvements['queue_length']['improvement_pct'] > 0
    wait_ok = improvements['wait_time']['improvement_pct'] > 0
    reward_ok = improvements['reward']['improvement_pct'] > 0
    
    print(f"  {'[✓]' if queue_ok else '[✗]'} Queue length reduced")
    print(f"  {'[✓]' if wait_ok else '[✗]'} Wait time reduced")
    print(f"  {'[✓]' if reward_ok else '[✗]'} Reward improved")
    
    success = queue_ok and wait_ok and reward_ok
    if success:
        print(f"\n{'='*100}")
        print("🎉 PPO MODEL BEATS BASELINE IN ALL METRICS! 🎉")
        print(f"{'='*100}\n")
    else:
        print(f"\n{'='*100}")
        print("⚠️  Analyzing results...")
        print(f"{'='*100}\n")


def save_evaluation(ppo_eval, filename="ppo_evaluation_v2.json"):
    """Save evaluation results to JSON"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w') as f:
        json.dump(ppo_eval, f, indent=2)
    
    print(f"\n[OK] Evaluation saved to {filename}")


def save_comparison(improvements, filename="comparison_results_v2.json"):
    """Save comparison results to JSON"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w') as f:
        json.dump(improvements, f, indent=2)
    
    print(f"[OK] Comparison saved to {filename}")


if __name__ == "__main__":
    print("\n" + "="*90)
    print("PPO MODEL EVALUATION - IMPROVED VERSION")
    print("="*90)
    
    # Load baseline
    print("\n[1/3] Loading baseline metrics...")
    baseline = load_baseline_metrics()
    if baseline is None:
        print("[ERROR] Cannot proceed without baseline metrics")
        exit(1)
    print("[OK] Baseline metrics loaded")
    
    # Evaluate PPO model
    print("\n[2/3] Evaluating PPO model...")
    # Update this path to test different model versions:
    # - v2: "models/ppo_model_v2/ppo_final_v2.zip" (baseline RL agent)
    # - v3: "models/ppo_model_v3/ppo_final_v3.zip" (ambulance priority agent)
    model_path = "models/ppo_model_v3/ppo_final_v3.zip"  # V3 - Ambulance Priority
    ppo_eval = evaluate_ppo_model(model_path, num_episodes=3, max_steps=3600)
    
    if ppo_eval is None:
        print("[ERROR] Model evaluation failed")
        print(f"[TIP] Make sure model exists at: {model_path}")
        exit(1)
    
    # Calculate improvements
    print("\n[3/3] Calculating improvements...")
    improvements = calculate_improvement(baseline, ppo_eval)
    print("[OK] Improvements calculated")
    
    # Print comparison
    print_comparison(baseline, ppo_eval, improvements)
    
    # Save results
    save_evaluation(ppo_eval)
    save_comparison(improvements)
    
    print("\n✓ Evaluation complete!")
    print(f"\nResults saved:")
    print(f"  - ppo_evaluation_v2.json (detailed metrics)")
    print(f"  - comparison_results_v2.json (improvement percentages)")
