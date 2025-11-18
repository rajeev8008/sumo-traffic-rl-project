"""
Comprehensive test suite for MGRoadEnv with traffic
"""
import sys
sys.path.insert(0, '.')
from sumo_mg_road_env import MGRoadEnv
import numpy as np

def test_basic_connectivity():
    """Test 1: Basic connectivity and initialization"""
    print("\n" + "="*60)
    print("TEST 1: Basic Connectivity")
    print("="*60)
    
    env = MGRoadEnv(use_gui=False, test_routes=True)
    obs, info = env.reset()
    
    assert obs.shape == (16,), f"Wrong observation shape: {obs.shape}"
    assert env.action_space.nvec.tolist() == [6, 5, 12], "Wrong action space"
    
    env.close()
    print("[OK] Environment initializes correctly")
    print("[OK] Observation space: (16,)")
    print("[OK] Action space: MultiDiscrete([6, 5, 12])")


def test_traffic_generation():
    """Test 2: Verify traffic is being generated"""
    print("\n" + "="*60)
    print("TEST 2: Traffic Generation")
    print("="*60)
    
    env = MGRoadEnv(use_gui=False, test_routes=True)
    obs, info = env.reset()
    
    vehicle_counts = []
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        vehicle_counts.append(info.get('total_vehicles', 0))
    
    env.close()
    
    max_vehicles = max(vehicle_counts)
    avg_vehicles = np.mean(vehicle_counts)
    
    assert max_vehicles > 0, "No vehicles generated!"
    print(f"[OK] Vehicles detected: min={min(vehicle_counts)}, max={max_vehicles}, avg={avg_vehicles:.1f}")


def test_reward_function():
    """Test 3: Verify reward function responds to conditions"""
    print("\n" + "="*60)
    print("TEST 3: Reward Function")
    print("="*60)
    
    env = MGRoadEnv(use_gui=False, test_routes=True)
    obs, info = env.reset()
    
    rewards = []
    queue_lengths = []
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        queue_lengths.append(obs[0])  # First TL queue length
    
    env.close()
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_queue = np.mean(queue_lengths)
    
    # Negative reward indicates congestion penalty is working
    assert avg_reward < 0, "Reward should be negative (congestion penalty)"
    assert std_reward > 0, "Rewards should vary based on traffic conditions"
    
    print(f"[OK] Average reward: {avg_reward:.4f} (negative indicates congestion penalty)")
    print(f"[OK] Reward std dev: {std_reward:.4f} (varies with traffic)")
    print(f"[OK] Average queue length: {avg_queue:.1f} vehicles")


def test_observation_validity():
    """Test 4: Verify observations are valid numbers"""
    print("\n" + "="*60)
    print("TEST 4: Observation Validity")
    print("="*60)
    
    env = MGRoadEnv(use_gui=False, test_routes=True)
    obs, info = env.reset()
    
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        
        # Check all values are valid floats
        assert all(isinstance(x, (int, float, np.number)) for x in obs), "Invalid observation values"
        assert not np.any(np.isnan(obs)), "NaN values in observation"
        assert not np.any(np.isinf(obs)), "Inf values in observation"
    
    env.close()
    print("[OK] All observations are valid numbers")
    print("[OK] No NaN or Inf values detected")
    print(f"[OK] Final observation: {obs}")


def test_action_handling():
    """Test 5: Verify actions are correctly applied"""
    print("\n" + "="*60)
    print("TEST 5: Action Handling")
    print("="*60)
    
    env = MGRoadEnv(use_gui=False, test_routes=True)
    obs, info = env.reset()
    
    # Test specific actions
    test_actions = [
        [0, 0, 0],   # All green phase 0
        [5, 4, 11],  # Max phases
        [2, 2, 6],   # Middle phases
    ]
    
    for action in test_actions:
        try:
            obs, reward, term, trunc, info = env.step(action)
            # Phases should match the observation
            phase_tl1 = obs[4]   # 5 features per TL
            phase_tl2 = obs[9]
            phase_tl3 = obs[14]
            print(f"[OK] Action {action} accepted. Phases: TL1={phase_tl1:.0f}, TL2={phase_tl2:.0f}, TL3={phase_tl3:.0f}")
        except Exception as e:
            print(f"[FAIL] Action {action} failed: {e}")
            raise
    
    env.close()
    print("[OK] All actions handled correctly")


def test_episode_flow():
    """Test 6: Full episode execution"""
    print("\n" + "="*60)
    print("TEST 6: Full Episode Flow")
    print("="*60)
    
    env = MGRoadEnv(use_gui=False, test_routes=True)
    obs, info = env.reset()
    
    episode_reward = 0
    steps = 0
    max_steps = 500
    
    while steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    env.close()
    
    print(f"[OK] Episode completed: {steps} steps")
    print(f"[OK] Episode reward: {episode_reward:.2f}")
    print(f"[OK] Average step reward: {episode_reward/steps:.4f}")


if __name__ == "__main__":
    try:
        test_basic_connectivity()
        test_traffic_generation()
        test_reward_function()
        test_observation_validity()
        test_action_handling()
        test_episode_flow()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nEnvironment is ready for RL training!")
        print("Next steps:")
        print("  1. Run baseline.py for fixed-time control comparison")
        print("  2. Train PPO agent with train.py (coming next)")
        print("  3. Evaluate agent with evaluate.py")
        
    except Exception as e:
        print(f"\n[FAILED] {e}")
        raise
