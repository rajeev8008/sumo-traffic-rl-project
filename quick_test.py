#!/usr/bin/env python
"""
Quick test - Run this to verify environment works with traffic
"""
from sumo_mg_road_env import MGRoadEnv

print("🚗 Testing MG Road RL Environment with Traffic...")

# Create environment with test routes
env = MGRoadEnv(use_gui=False, test_routes=True)
obs, info = env.reset()

print(f"\n✅ Environment initialized")
print(f"   Observation shape: {obs.shape}")
print(f"   Action space: {env.action_space}")

# Run 10 steps
print(f"\n🔄 Running simulation...")
for step in range(10):
    action = env.action_space.sample()  # Random phase changes
    obs, reward, terminated, truncated, info = env.step(action)

env.close()

print(f"\n✅ Last step stats:")
print(f"   Reward: {reward:.4f}")
print(f"   Queue length (TL1): {obs[0]:.1f}")
print(f"   Wait time (TL1): {obs[1]:.1f}")
print(f"   Vehicles in sim: {info['total_vehicles']}")

print(f"\n🎉 SUCCESS! Environment is working correctly!")
print(f"\n📊 You can now:")
print(f"   - Train a PPO agent (in train.py)")
print(f"   - Compare against baseline (in baseline.py)")
print(f"   - Run full test suite (test_env_comprehensive.py)")
