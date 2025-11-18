"""
Cross-map evaluation script for trained PPO agent.

This script evaluates the best_model on different SUMO configurations
to detect overfitting and compare performance metrics across maps.

Usage:
    python evaluate_cross_map.py --model ./models/ppo_mg_road/best_model.zip
    python evaluate_cross_map.py --maps mg_road osm --episodes 5
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

from SumoEnv import SumoEnv
from ppo_agent import load_ppo_agent


class CrossMapEvaluationConfig:
    """Configuration for cross-map evaluation"""
    
    # Map configurations
    MAPS = {
        "mg_road": "SUMO_Trinity_Traffic_sim/osm.sumocfg",
        "osm": "osm_sudo_map_2/osm.sumocfg",
    }
    
    USE_GUI = False
    MAX_EPISODE_STEPS = 120  # Match training environment
    
    # Evaluation seeds - diverse seeds to test generalization
    EVAL_SEEDS = [42, 100, 200, 300, 400]  # Include training seed (42)
    N_EPISODES_PER_SEED = 3
    
    # Output
    EVAL_LOG_DIR = "./logs/cross_map_evaluation"


def create_eval_env(sumocfg_file: str, seed: int = None) -> SumoEnv:
    """Create evaluation environment with specified SUMO configuration."""
    env = SumoEnv(
        use_gui=CrossMapEvaluationConfig.USE_GUI,
        sumocfg_file=sumocfg_file,
    )
    if seed is not None:
        env.action_space.seed(seed)
    return env


def run_episode(
    agent,
    env: SumoEnv,
    deterministic: bool = True,
    max_steps: int = CrossMapEvaluationConfig.MAX_EPISODE_STEPS,
) -> Dict:
    """Run a single episode and collect metrics."""
    obs, _ = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    rewards = []
    actions = []
    
    while not done and steps < max_steps:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        rewards.append(float(reward))
        actions.append(int(action))
        steps += 1
    
    return {
        "total_reward": float(episode_reward),
        "steps": steps,
        "avg_step_reward": float(episode_reward / steps) if steps > 0 else 0.0,
        "rewards": rewards,
        "actions": actions,
        "action_diversity": float(np.mean(actions)) if actions else 0.0,
    }


def evaluate_on_seed(
    agent,
    sumocfg_file: str,
    seed: int,
    n_episodes: int = 3,
    deterministic: bool = True,
) -> Dict:
    """Evaluate agent on a specific seed for N episodes."""
    print(f"      Seed {seed:3d} ({n_episodes} episodes)...", end=" ", flush=True)
    env = create_eval_env(sumocfg_file=sumocfg_file, seed=seed)
    
    max_steps = env.max_episode_steps if hasattr(env, 'max_episode_steps') else CrossMapEvaluationConfig.MAX_EPISODE_STEPS
    
    episode_results = []
    for ep in range(n_episodes):
        result = run_episode(agent, env, deterministic=deterministic, max_steps=max_steps)
        episode_results.append(result)
    
    env.close()
    
    # Aggregate statistics
    total_rewards = [r["total_reward"] for r in episode_results]
    avg_step_rewards = [r["avg_step_reward"] for r in episode_results]
    action_diversities = [r["action_diversity"] for r in episode_results]
    
    summary = {
        "seed": seed,
        "n_episodes": n_episodes,
        "episodes": episode_results,
        "mean_total_reward": float(np.mean(total_rewards)),
        "std_total_reward": float(np.std(total_rewards)),
        "min_total_reward": float(np.min(total_rewards)),
        "max_total_reward": float(np.max(total_rewards)),
        "mean_step_reward": float(np.mean(avg_step_rewards)),
        "std_step_reward": float(np.std(avg_step_rewards)),
        "mean_action_diversity": float(np.mean(action_diversities)),
    }
    
    print(f"Mean Reward: {summary['mean_total_reward']:8.3f} ± {summary['std_total_reward']:6.3f}")
    
    return summary


def evaluate_map(
    agent,
    map_name: str,
    sumocfg_file: str,
    eval_seeds: List[int] = None,
    n_episodes: int = 3,
) -> Dict:
    """Evaluate agent on a specific map across multiple seeds."""
    if eval_seeds is None:
        eval_seeds = CrossMapEvaluationConfig.EVAL_SEEDS
    
    print(f"\n   Evaluating on map: {map_name}")
    print(f"   Config: {sumocfg_file}")
    print(f"   Seeds: {eval_seeds}")
    print()
    
    seed_results = []
    for seed in eval_seeds:
        result = evaluate_on_seed(
            agent,
            sumocfg_file=sumocfg_file,
            seed=seed,
            n_episodes=n_episodes,
            deterministic=True,
        )
        seed_results.append(result)
    
    # Aggregate across all seeds
    all_rewards = [r["mean_total_reward"] for r in seed_results]
    all_step_rewards = [r["mean_step_reward"] for r in seed_results]
    
    map_summary = {
        "map_name": map_name,
        "sumocfg_file": sumocfg_file,
        "n_seeds": len(eval_seeds),
        "n_episodes_per_seed": n_episodes,
        "seed_results": seed_results,
        "overall_mean_reward": float(np.mean(all_rewards)),
        "overall_std_reward": float(np.std(all_rewards)),
        "overall_min_reward": float(np.min(all_rewards)),
        "overall_max_reward": float(np.max(all_rewards)),
        "overall_mean_step_reward": float(np.mean(all_step_rewards)),
        "overall_std_step_reward": float(np.std(all_step_rewards)),
    }
    
    return map_summary


def compute_overfitting_metrics(training_map_result: Dict, test_map_result: Dict) -> Dict:
    """
    Compute overfitting metrics by comparing training and test map performance.
    
    Overfitting occurs when training_map >> test_map performance.
    """
    train_reward = training_map_result["overall_mean_reward"]
    test_reward = test_map_result["overall_mean_reward"]
    
    # Compute performance drop
    reward_drop = train_reward - test_reward
    
    # Compute percentage drop (relative to absolute value of training reward)
    if train_reward != 0:
        reward_drop_pct = (reward_drop / abs(train_reward)) * 100
    else:
        reward_drop_pct = 0.0
    
    # Overfitting threshold: > 15% drop in reward
    overfitting_threshold_pct = 15.0
    is_overfitted = reward_drop_pct > overfitting_threshold_pct
    
    return {
        "training_map": training_map_result["map_name"],
        "test_map": test_map_result["map_name"],
        "training_map_reward": float(train_reward),
        "test_map_reward": float(test_reward),
        "reward_drop": float(reward_drop),
        "reward_drop_pct": float(reward_drop_pct),
        "overfitting_threshold_pct": overfitting_threshold_pct,
        "is_overfitted": bool(is_overfitted),
    }


def full_cross_map_evaluation(model_path: str, maps_to_eval: List[str] = None) -> Dict:
    """
    Run full cross-map evaluation.
    
    Assumes the first map in maps_to_eval is the training map.
    """
    if maps_to_eval is None:
        maps_to_eval = list(CrossMapEvaluationConfig.MAPS.keys())
    
    print("\n" + "=" * 80)
    print("[CROSS-MAP EVALUATION FOR OVERFITTING DETECTION]")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Maps to evaluate: {maps_to_eval}")
    print(f"Evaluation seeds: {CrossMapEvaluationConfig.EVAL_SEEDS}")
    print(f"Episodes per seed: {CrossMapEvaluationConfig.N_EPISODES_PER_SEED}")
    print("=" * 80)
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\nLoading model: {model_path}")
    agent = load_ppo_agent(model_path)
    print("✓ Model loaded successfully!")
    
    # Create output directory
    os.makedirs(CrossMapEvaluationConfig.EVAL_LOG_DIR, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "maps": maps_to_eval,
    }
    
    # Evaluate on each map
    map_results = {}
    for i, map_name in enumerate(maps_to_eval):
        if map_name not in CrossMapEvaluationConfig.MAPS:
            print(f"\n⚠️  Unknown map: {map_name}. Skipping.")
            continue
        
        sumocfg_file = CrossMapEvaluationConfig.MAPS[map_name]
        print(f"\n[{i+1}/{len(maps_to_eval)}] Evaluating map: {map_name}")
        
        map_result = evaluate_map(
            agent,
            map_name=map_name,
            sumocfg_file=sumocfg_file,
            eval_seeds=CrossMapEvaluationConfig.EVAL_SEEDS,
            n_episodes=CrossMapEvaluationConfig.N_EPISODES_PER_SEED,
        )
        map_results[map_name] = map_result
    
    results["map_results"] = map_results
    
    # Compute overfitting metrics (training map vs other maps)
    if len(map_results) > 1:
        training_map = maps_to_eval[0]  # Assume first map is training map
        if training_map in map_results:
            print("\n" + "=" * 80)
            print("[OVERFITTING ANALYSIS]")
            print("=" * 80)
            print(f"Training map: {training_map}")
            
            overfitting_results = {}
            for test_map in maps_to_eval[1:]:
                if test_map in map_results:
                    metrics = compute_overfitting_metrics(
                        map_results[training_map],
                        map_results[test_map],
                    )
                    overfitting_results[test_map] = metrics
                    
                    print(f"\n{training_map} vs {test_map}:")
                    print(f"  Training ({training_map}) reward: {metrics['training_map_reward']:8.3f}")
                    print(f"  Test ({test_map}) reward:      {metrics['test_map_reward']:8.3f}")
                    print(f"  Reward drop: {metrics['reward_drop']:8.3f} ({metrics['reward_drop_pct']:6.2f}%)")
                    
                    if metrics["is_overfitted"]:
                        print(f"  ⚠️  OVERFITTING DETECTED (drop > {metrics['overfitting_threshold_pct']:.0f}%)")
                    else:
                        print(f"  ✓ Model generalizes well (drop ≤ {metrics['overfitting_threshold_pct']:.0f}%)")
            
            results["overfitting_analysis"] = overfitting_results
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("[PERFORMANCE SUMMARY BY MAP]")
    print("=" * 80)
    for map_name, map_result in map_results.items():
        print(f"\n{map_name}:")
        print(f"  Overall reward: {map_result['overall_mean_reward']:8.3f} ± {map_result['overall_std_reward']:6.3f}")
        print(f"  Min/Max: {map_result['overall_min_reward']:8.3f} / {map_result['overall_max_reward']:8.3f}")
        print(f"  Step reward: {map_result['overall_mean_step_reward']:8.3f} ± {map_result['overall_std_step_reward']:6.3f}")
    
    # Save results to JSON
    eval_results_path = os.path.join(
        CrossMapEvaluationConfig.EVAL_LOG_DIR,
        f"cross_map_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {eval_results_path}")
    print("=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO agent across multiple maps for overfitting detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/ppo_mg_road/best_model.zip",
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--maps",
        type=str,
        nargs="+",
        default=["mg_road", "osm"],
        help="Maps to evaluate on (in order: training_map, test_map1, test_map2, ...)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodes per seed for evaluation",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=CrossMapEvaluationConfig.EVAL_SEEDS,
        help="Random seeds to evaluate on",
    )
    
    args = parser.parse_args()
    
    # Update config
    CrossMapEvaluationConfig.N_EPISODES_PER_SEED = args.episodes
    CrossMapEvaluationConfig.EVAL_SEEDS = args.seeds
    
    try:
        results = full_cross_map_evaluation(args.model, maps_to_eval=args.maps)
    except Exception as e:
        print(f"\n[ERROR] Cross-map evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
