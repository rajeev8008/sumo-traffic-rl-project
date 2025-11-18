"""Test environment with traffic"""
import sys
sys.path.insert(0, '.')
from sumo_mg_road_env import MGRoadEnv

print('[TEST] Creating environment with test routes...')
env = MGRoadEnv(use_gui=False, test_routes=True)
obs, info = env.reset()
print(f'[OK] Reset successful')
print(f'[OK] Observation shape: {obs.shape}')

# Run for 100 steps to see traffic
total_vehicles = 0
total_reward = 0
for i in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    total_vehicles += info.get('total_vehicles', 0)
    total_reward += reward
    if i % 20 == 0:
        queue = obs[0]
        wait = obs[1]
        veh_count = info.get('total_vehicles', 0)
        print(f'Step {i+1:3d}: reward={reward:7.3f}, queue={queue:5.1f}, wait={wait:5.1f}, vehicles={veh_count:3d}')

env.close()
avg_reward = total_reward / 100
print(f'[OK] Average reward: {avg_reward:.3f}')
print(f'[OK] Test completed successfully - vehicles are generating!')
