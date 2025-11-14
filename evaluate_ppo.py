"""
Evaluation script for trained PPO agent on mg_road.

This script evaluates the trained model for:
1. Performance on the training environment (mg_road)
2. Generalization to other random seeds and scenarios
3. Overfitting detection
4. Deterministic vs stochastic policy comparison
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
import argparse
from datetime import datetime

# Assuming SumoEnv and load_ppo_agent are correctly imported from your modules
from SumoEnv import SumoEnv
from ppo_agent import load_ppo_agent


class EvaluationConfig:
    """Configuration for evaluation"""
    
    SUMOCFG_FILE = "SUMO_Trinity_Traffic_sim/osm.sumocfg"
    USE_GUI = False
    
    # CRITICAL CHANGE: Set Max Steps to match the optimized training environment (120 agent steps)
    # This must match the constant used in SumoEnv.py: MAX_EPISODE_STEPS_TRAINING
    MAX_EPISODE_STEPS = 120 
    
    # Evaluation seeds - use different seeds than training to test generalization
    EVAL_SEEDS = [100, 200, 300, 400, 500]
    N_EPISODES_PER_SEED = 3
    
    # For overfitting detection
    TRAIN_SEED = 42  # Should match training seed
    
    # Output
    EVAL_LOG_DIR = "./logs/ppo_evaluation"


def create_eval_env(seed: int = None) -> SumoEnv:
    """Create evaluation environment."""
    env = SumoEnv(
        use_gui=EvaluationConfig.USE_GUI,
        sumocfg_file=EvaluationConfig.SUMOCFG_FILE,
    )
    if seed is not None:
        # Note: Since SumoEnv.reset() sets the SUMO seed, this action_space.seed() 
        # primarily seeds the gymnasium action sampler, which is less relevant for deterministic evaluation.
        env.action_space.seed(seed)
    return env


def run_episode(
    agent,
    env: SumoEnv,
    deterministic: bool = True,
    # MAX_STEPS is now derived from config/env
    max_steps: int = EvaluationConfig.MAX_EPISODE_STEPS, 
) -> dict:
    """
    Run a single episode and collect metrics.
    """
    obs, _ = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    rewards = []
    actions = []
    
    # Use max_steps from config/argument
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
        "action_diversity": float(np.mean(actions)),  # Avg action (0 or 1)
    }


def evaluate_on_seed(
    agent,
    seed: int,
    n_episodes: int = 3,
    deterministic: bool = True,
) -> dict:
    """
    Evaluate agent on a specific seed for N episodes.
    """
    print(f"\n   Evaluating seed {seed} ({n_episodes} episodes)...")
    env = create_eval_env(seed=seed)
    
    # Set max steps based on the environment's configured length
    max_steps = env.max_episode_steps if hasattr(env, 'max_episode_steps') else EvaluationConfig.MAX_EPISODE_STEPS
    
    episode_results = []
    
    for ep in range(n_episodes):
        # Pass the derived max_steps to run_episode
        result = run_episode(agent, env, deterministic=deterministic, max_steps=max_steps) 
        episode_results.append(result)
        print(f"     Episode {ep + 1}: Reward={result['total_reward']:.3f}, "
              f"Steps={result['steps']}, Action_Diversity={result['action_diversity']:.3f}")
    
    env.close()
    
    # Aggregate statistics
    total_rewards = [r["total_reward"] for r in episode_results]
    avg_step_rewards = [r["avg_step_reward"] for r in episode_results]
    action_diversities = [r["action_diversity"] for r in episode_results]
    
    return {
        "seed": seed,
        "episodes": episode_results,
        "mean_total_reward": float(np.mean(total_rewards)),
        "std_total_reward": float(np.std(total_rewards)),
        "min_total_reward": float(np.min(total_rewards)),
        "max_total_reward": float(np.max(total_rewards)),
        "mean_step_reward": float(np.mean(avg_step_rewards)),
        "mean_action_diversity": float(np.mean(action_diversities)),
    }


def evaluate_deterministic_vs_stochastic(
    agent,
    seed: int = 42,
    n_episodes: int = 3,
) -> dict:
    """
    Compare deterministic vs stochastic policy to assess confidence.
    """
    print("\n   Comparing deterministic vs stochastic policy...")
    
    det_results = []
    stoch_results = []
    
    # Determine max steps from a dummy environment instance
    temp_env = create_eval_env(seed=seed)
    max_steps = temp_env.max_episode_steps if hasattr(temp_env, 'max_episode_steps') else EvaluationConfig.MAX_EPISODE_STEPS
    temp_env.close()
    
    for ep in range(n_episodes):
        # Re-create envs inside the loop to ensure fresh resets
        det_result = run_episode(agent, create_eval_env(seed=seed), deterministic=True, max_steps=max_steps)
        stoch_result = run_episode(agent, create_eval_env(seed=seed), deterministic=False, max_steps=max_steps)
        
        det_results.append(det_result["total_reward"])
        stoch_results.append(stoch_result["total_reward"])
        
        # Manually close temporary environments created in the loop
        # (This is handled by the calls to create_eval_env, but explicitly showing the intent)
        
    return {
        "deterministic_mean": float(np.mean(det_results)),
        "deterministic_std": float(np.std(det_results)),
        "stochastic_mean": float(np.mean(stoch_results)),
        "stochastic_std": float(np.std(stoch_results)),
        "difference": float(np.mean(det_results) - np.mean(stoch_results)),
    }


def detect_overfitting(
    agent,
    train_seed: int = 42,
    eval_seeds: list = None,
    n_episodes: int = 3,
) -> dict:
    """
    Detect overfitting by comparing performance on training seed vs other seeds.
    """
    if eval_seeds is None:
        eval_seeds = [100, 200, 300]
    
    print("\n" + "=" * 70)
    print("[OVERFITTING DETECTION]")
    print("=" * 70)
    
    # Performance on training seed
    # Use a set of seeds that should match your training setup (e.g., TRAIN_SEED)
    print(f"\nEvaluating on TRAINING seed ({train_seed})...")
    train_perf = evaluate_on_seed(agent, train_seed, n_episodes=n_episodes)
    train_reward = train_perf["mean_total_reward"]
    
    # Filter out the training seed from the generalization seeds
    other_eval_seeds = [s for s in eval_seeds if s != train_seed]
    
    print(f"\nEvaluating on OTHER seeds...")
    other_perfs = []
    for seed in other_eval_seeds:
        perf = evaluate_on_seed(agent, seed, n_episodes=n_episodes)
        other_perfs.append(perf)
    
    other_rewards = [p["mean_total_reward"] for p in other_perfs]
    other_reward_mean = np.mean(other_rewards)
    other_reward_std = np.std(other_rewards)
    
    # Compute overfitting metrics (assuming more negative reward is WORSE)
    # A positive difference means the training seed performed better (lower penalty)
    perf_drop = train_reward - other_reward_mean
    perf_drop_pct = (perf_drop / abs(train_reward) * 100) if train_reward != 0 else 0.0
    
    print("\n" + "-" * 70)
    print(f"Training seed performance: {train_reward:.3f}")
    print(f"Other seeds avg performance: {other_reward_mean:.3f} ± {other_reward_std:.3f}")
    print(f"Performance drop (Train - Other): {perf_drop:.3f} ({perf_drop_pct:.1f}%)")
    
    overfitting_threshold = 0.1  # 10% drop
    is_overfitted = perf_drop_pct > (overfitting_threshold * 100)
    
    if is_overfitted:
        print(f"⚠️  WARNING: Model appears to be OVERFITTED (drop > {overfitting_threshold*100:.0f}%)")
    else:
        print(f"✓ Model generalizes well (drop <= {overfitting_threshold*100:.0f}%)")
    
    return {
        "train_seed": train_seed,
        "train_performance": train_perf,
        "other_seeds_performance": other_perfs,
        "train_reward": float(train_reward),
        "other_seeds_avg": float(other_reward_mean),
        "other_seeds_std": float(other_reward_std),
        "performance_drop": float(perf_drop),
        "performance_drop_pct": float(perf_drop_pct),
        "is_overfitted": bool(is_overfitted),
    }


def evaluate_generalization(
    agent,
    eval_seeds: list = None,
    n_episodes: int = 3,
) -> dict:
    """
    Evaluate generalization across multiple seeds.
    """
    if eval_seeds is None:
        eval_seeds = EvaluationConfig.EVAL_SEEDS
    
    print("\n" + "=" * 70)
    print("[GENERALIZATION EVALUATION]")
    print("=" * 70)
    
    results = []
    for seed in eval_seeds:
        perf = evaluate_on_seed(agent, seed, n_episodes=n_episodes)
        results.append(perf)
    
    all_rewards = [p["mean_total_reward"] for p in results]
    
    print(f"\nGeneralization Summary:")
    print(f"  Mean across seeds: {np.mean(all_rewards):.3f}")
    print(f"  Std across seeds: {np.std(all_rewards):.3f}")
    print(f"  Min: {np.min(all_rewards):.3f}")
    print(f"  Max: {np.max(all_rewards):.3f}")
    print(f"  Range: {np.max(all_rewards) - np.min(all_rewards):.3f}")
    
    return {
        "per_seed_results": results,
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "min_reward": float(np.min(all_rewards)),
        "max_reward": float(np.max(all_rewards)),
    }


def full_evaluation(model_path: str) -> dict:
    """
    Run full evaluation suite on a trained model.
    """
    print("\n" + "=" * 70)
    print("[PPO AGENT EVALUATION]")
    print("=" * 70)
    print(f"[LOAD] Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    agent = load_ppo_agent(model_path)
    print("[LOAD] Model loaded successfully!")
    
    # Create output directory
    os.makedirs(EvaluationConfig.EVAL_LOG_DIR, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
    }
    
    # 1. Overfitting detection
    overfitting_analysis = detect_overfitting(
        agent,
        train_seed=EvaluationConfig.TRAIN_SEED,
        eval_seeds=EvaluationConfig.EVAL_SEEDS,
        n_episodes=EvaluationConfig.N_EPISODES_PER_SEED,
    )
    results["overfitting_analysis"] = overfitting_analysis
    
    # 2. Generalization evaluation
    # Use the seeds that performed poorly in the overfitting check for generalization analysis
    generalization_results = evaluate_generalization(
        agent,
        eval_seeds=EvaluationConfig.EVAL_SEEDS,
        n_episodes=EvaluationConfig.N_EPISODES_PER_SEED,
    )
    results["generalization"] = generalization_results
    
    # 3. Deterministic vs Stochastic
    print("\n" + "=" * 70)
    print("[POLICY ANALYSIS]")
    print("=" * 70)
    policy_analysis = evaluate_deterministic_vs_stochastic(
        agent,
        seed=EvaluationConfig.TRAIN_SEED,
        n_episodes=3,
    )
    results["policy_analysis"] = policy_analysis
    print(f"Deterministic mean: {policy_analysis['deterministic_mean']:.3f}")
    print(f"Stochastic mean: {policy_analysis['stochastic_mean']:.3f}")
    print(f"Difference (Det - Stoch): {policy_analysis['difference']:.3f}")
    
    # Save results
    eval_results_path = os.path.join(EvaluationConfig.EVAL_LOG_DIR, "evaluation_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Evaluation results saved to {eval_results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("[EVALUATION SUMMARY]")
    print("=" * 70)
    print(f"Overfitting Status: {'YES ⚠️' if overfitting_analysis['is_overfitted'] else 'NO ✓'}")
    print(f"Performance on training seed: {overfitting_analysis['train_reward']:.3f}")
    print(f"Average performance on other seeds: {overfitting_analysis['other_seeds_avg']:.3f}")
    print(f"Generalization (Std across seeds): {generalization_results['std_reward']:.3f}")
    print("=" * 70 + "\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument(
        "--model",
        type=str,
        default="./models/ppo_mg_road/best_model.zip",
        help="Path to trained model (.zip file)",
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
        default=EvaluationConfig.EVAL_SEEDS,
        help="Random seeds to evaluate on",
    )
    
    args = parser.parse_args()
    
    # Update config
    EvaluationConfig.N_EPISODES_PER_SEED = args.episodes
    EvaluationConfig.EVAL_SEEDS = args.seeds
    
    # Run evaluation
    try:
        results = full_evaluation(args.model)
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)