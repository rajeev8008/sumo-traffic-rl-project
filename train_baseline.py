"""
Baseline Agent - Fixed-Time Traffic Light Controller

Evaluates performance of a simple fixed-time controller:
- Each traffic light cycles through phases at fixed intervals
- Phase duration: 40 seconds (configurable)
- Provides baseline metrics for RL agent comparison

Metrics Collected:
- Average queue length per episode
- Average wait time per episode
- Maximum vehicles in system
- Total episode reward
- Episode completion time
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
sys.path.insert(0, '.')

from sumo_mg_road_env import MGRoadEnv


class FixedTimeController:
    """
    Fixed-time traffic light controller
    
    Each traffic light cycles through its phases at a fixed duration
    """
    
    def __init__(self, phase_duration=40):
        """
        Initialize fixed-time controller
        
        Args:
            phase_duration (int): Duration of each phase in seconds
        """
        self.phase_duration = phase_duration
        self.elapsed_times = [0, 0, 0]  # Track elapsed time per TL
    
    def get_action(self, observation, elapsed_time):
        """
        Compute next action based on elapsed time
        
        Args:
            observation: Current observation (not used by fixed-time)
            elapsed_time: Simulation time in seconds
        
        Returns:
            action: [phase_tl1, phase_tl2, phase_tl3]
        """
        # TL1: 6 phases
        phase_tl1 = int((elapsed_time / self.phase_duration) % 6)
        
        # TL2: 5 phases
        phase_tl2 = int((elapsed_time / self.phase_duration) % 5)
        
        # TL3: 12 phases
        phase_tl3 = int((elapsed_time / self.phase_duration) % 12)
        
        return [phase_tl1, phase_tl2, phase_tl3]


def get_vehicle_type_stats(env):
    """
    Get statistics for each vehicle type
    
    Returns:
        dict: Vehicle type statistics with keys like 'motorcycle', 'car', etc.
    """
    import traci
    
    # Initialize vehicle type counters
    vehicle_types = {
        'motorcycle': {'count': 0, 'wait_times': []},
        'car': {'count': 0, 'wait_times': []},
        'taxi': {'count': 0, 'wait_times': []},
        'bus': {'count': 0, 'wait_times': []},
        'truck': {'count': 0, 'wait_times': []},
        'ambulance': {'count': 0, 'wait_times': []},
        'emergency': {'count': 0, 'wait_times': []},
        'fire': {'count': 0, 'wait_times': []},
        'other': {'count': 0, 'wait_times': []},
    }
    
    # Iterate through all vehicles and collect stats
    try:
        all_vehicles = traci.vehicle.getIDList()
        for veh in all_vehicles:
            try:
                vtype = traci.vehicle.getTypeID(veh)
                wait_time = traci.vehicle.getWaitingTime(veh)
                
                # Classify vehicle
                if vtype.startswith('motorcycle'):
                    vehicle_types['motorcycle']['count'] += 1
                    vehicle_types['motorcycle']['wait_times'].append(wait_time)
                elif vtype.startswith('car'):
                    vehicle_types['car']['count'] += 1
                    vehicle_types['car']['wait_times'].append(wait_time)
                elif vtype.startswith('taxi'):
                    vehicle_types['taxi']['count'] += 1
                    vehicle_types['taxi']['wait_times'].append(wait_time)
                elif vtype.startswith(('bus', 'coach')):
                    vehicle_types['bus']['count'] += 1
                    vehicle_types['bus']['wait_times'].append(wait_time)
                elif vtype.startswith('truck'):
                    vehicle_types['truck']['count'] += 1
                    vehicle_types['truck']['wait_times'].append(wait_time)
                elif vtype.startswith('ambulance'):
                    vehicle_types['ambulance']['count'] += 1
                    vehicle_types['ambulance']['wait_times'].append(wait_time)
                elif vtype.startswith(('emergency', 'fire')):
                    vehicle_types[vtype]['count'] += 1
                    vehicle_types[vtype]['wait_times'].append(wait_time)
                else:
                    vehicle_types['other']['count'] += 1
                    vehicle_types['other']['wait_times'].append(wait_time)
            except:
                pass
    except:
        pass
    
    # Compute statistics for each type
    stats = {}
    for vtype, data in vehicle_types.items():
        if data['count'] > 0:
            stats[vtype] = {
                'count': data['count'],
                'avg_wait_time': float(np.mean(data['wait_times'])),
                'max_wait_time': float(np.max(data['wait_times'])),
                'min_wait_time': float(np.min(data['wait_times'])),
            }
    
    return stats


def run_baseline(num_episodes=3, max_steps=500, use_gui=False):
    """
    Run baseline agent for evaluation
    
    Args:
        num_episodes (int): Number of episodes to run
        max_steps (int): Max steps per episode
        use_gui (bool): Whether to use SUMO GUI
    
    Returns:
        dict: Baseline metrics
    """
    import traci
    
    # Initialize environment and controller
    env = MGRoadEnv(use_gui=use_gui, test_routes=True)
    controller = FixedTimeController(phase_duration=40)
    
    # Metrics collection
    all_episode_metrics = []
    
    print("\n" + "="*70)
    print("BASELINE AGENT - Fixed-Time Traffic Light Controller")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Max steps per episode: {max_steps}")
    print(f"  - Phase duration: 40 seconds")
    print(f"  - Use test routes (immediate traffic): True")
    print(f"  - GUI: {use_gui}")
    print("="*70)
    
    for episode in range(num_episodes):
        print(f"\n[EPISODE {episode + 1}/{num_episodes}]")
        
        # Reset environment
        obs, info = env.reset()
        
        # Episode metrics
        episode_reward = 0
        episode_steps = 0
        queue_lengths = []
        wait_times = []
        vehicle_counts = []
        all_vehicle_types = {
            'motorcycle': {'counts': [], 'wait_times': []},
            'car': {'counts': [], 'wait_times': []},
            'auto': {'counts': [], 'wait_times': []},
            'bus': {'counts': [], 'wait_times': []},
            'truck': {'counts': [], 'wait_times': []},
            'ambulance': {'counts': [], 'wait_times': []},
        }
        
        # Run episode
        for step in range(max_steps):
            # Get current simulation time
            elapsed_time = obs[-1]  # Last element is sim_time
            
            # Compute action using fixed-time controller
            action = controller.get_action(obs, elapsed_time)
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Collect metrics
            episode_reward += reward
            episode_steps += 1
            queue_lengths.append(obs[0])  # TL1 queue length
            wait_times.append(obs[1])     # TL1 wait time
            vehicle_counts.append(info['total_vehicles'])
            
            # Collect per-vehicle-type statistics
            try:
                all_vehicles = traci.vehicle.getIDList()
                ambulance_count = 0
                for veh in all_vehicles:
                    try:
                        vtype = traci.vehicle.getTypeID(veh)
                        wait_time = traci.vehicle.getWaitingTime(veh)
                        
                        if vtype.startswith('motorcycle'):
                            all_vehicle_types['motorcycle']['wait_times'].append(wait_time)
                        elif vtype.startswith('car'):
                            all_vehicle_types['car']['wait_times'].append(wait_time)
                        elif vtype.startswith('auto'):
                            all_vehicle_types['auto']['wait_times'].append(wait_time)
                        elif vtype.startswith(('bus', 'coach')):
                            all_vehicle_types['bus']['wait_times'].append(wait_time)
                        elif vtype.startswith('truck'):
                            all_vehicle_types['truck']['wait_times'].append(wait_time)
                        elif vtype.startswith('ambulance'):
                            ambulance_count += 1
                            all_vehicle_types['ambulance']['wait_times'].append(wait_time)
                    except:
                        pass
            except:
                pass
            
            if (step + 1) % 100 == 0:
                print(f"  Step {step + 1}/{max_steps}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Vehicles={info['total_vehicles']}, "
                      f"Avg Queue={np.mean(queue_lengths[-10:]):.1f}")
            
            if terminated or truncated:
                break
        
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
        }
        
        # Add per-vehicle-type metrics
        vehicle_type_stats = {}
        for vtype, data in all_vehicle_types.items():
            if data['wait_times']:
                vehicle_type_stats[vtype] = {
                    'avg_wait_time': float(np.mean(data['wait_times'])),
                    'max_wait_time': float(np.max(data['wait_times'])),
                    'min_wait_time': float(np.min(data['wait_times'])),
                    'count': len(set(data['wait_times']))  # Approximate count of samples
                }
        
        episode_metrics['vehicle_types'] = vehicle_type_stats
        all_episode_metrics.append(episode_metrics)
        
        print(f"\n  Episode Reward: {episode_metrics['total_reward']:.2f}")
        print(f"  Avg Queue Length: {episode_metrics['avg_queue_length']:.1f}")
        print(f"  Avg Wait Time: {episode_metrics['avg_wait_time']:.1f} sec")
        print(f"  Avg Vehicles: {episode_metrics['avg_vehicles']:.0f}")
        print(f"\n  Per-Vehicle-Type Metrics:")
        for vtype, stats in vehicle_type_stats.items():
            print(f"    {vtype.upper()}:")
            print(f"      - Avg Wait: {stats['avg_wait_time']:.1f}s | "
                  f"Max Wait: {stats['max_wait_time']:.1f}s | "
                  f"Min Wait: {stats['min_wait_time']:.1f}s")
    
    env.close()
    
    # Compute aggregate statistics
    baseline_metrics = {
        "controller_type": "FixedTimeController",
        "phase_duration": 40,
        "num_episodes": num_episodes,
        "timestamp": datetime.now().isoformat(),
        "episodes": all_episode_metrics,
        "aggregate": {
            "avg_reward": float(np.mean([m['total_reward'] for m in all_episode_metrics])),
            "std_reward": float(np.std([m['total_reward'] for m in all_episode_metrics])),
            "avg_queue_length": float(np.mean([m['avg_queue_length'] for m in all_episode_metrics])),
            "avg_wait_time": float(np.mean([m['avg_wait_time'] for m in all_episode_metrics])),
            "avg_vehicles": float(np.mean([m['avg_vehicles'] for m in all_episode_metrics])),
            "max_vehicles": int(np.max([m['max_vehicles'] for m in all_episode_metrics])),
        }
    }
    
    # Compute per-vehicle-type aggregate statistics
    vehicle_type_aggregate = {}
    vehicle_types = ['motorcycle', 'car', 'auto', 'bus', 'truck', 'ambulance']
    
    for vtype in vehicle_types:
        wait_times = []
        for episode in all_episode_metrics:
            if 'vehicle_types' in episode and vtype in episode['vehicle_types']:
                # Extract individual wait times from all samples
                stats = episode['vehicle_types'][vtype]
                wait_times.append(stats['avg_wait_time'])
        
        if wait_times:
            vehicle_type_aggregate[vtype] = {
                'avg_wait_time': float(np.mean(wait_times)),
                'max_wait_time': float(max([ep['vehicle_types'][vtype]['max_wait_time'] 
                                           for ep in all_episode_metrics 
                                           if 'vehicle_types' in ep and vtype in ep['vehicle_types']])),
                'min_wait_time': float(min([ep['vehicle_types'][vtype]['min_wait_time'] 
                                           for ep in all_episode_metrics 
                                           if 'vehicle_types' in ep and vtype in ep['vehicle_types']])),
            }
    
    baseline_metrics['aggregate']['vehicle_types'] = vehicle_type_aggregate
    
    return baseline_metrics


def save_baseline_metrics(metrics, filename="baseline_metrics.json"):
    """
    Save baseline metrics to JSON file
    
    Args:
        metrics (dict): Baseline metrics dictionary
        filename (str): Output filename
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] Baseline metrics saved to {filename}")


def print_baseline_summary(metrics):
    """
    Print summary of baseline metrics with per-vehicle-type breakdown
    
    Args:
        metrics (dict): Baseline metrics dictionary
    """
    agg = metrics['aggregate']
    
    print("\n" + "="*70)
    print("BASELINE SUMMARY")
    print("="*70)
    print(f"Episodes Run: {metrics['num_episodes']}")
    print(f"Controller: {metrics['controller_type']}")
    print(f"Phase Duration: {metrics['phase_duration']} sec")
    print()
    print(f"Aggregate Results (Average across all episodes):")
    print(f"  Total Reward: {agg['avg_reward']:.2f} (±{agg['std_reward']:.2f})")
    print(f"  Queue Length: {agg['avg_queue_length']:.1f} vehicles")
    print(f"  Wait Time: {agg['avg_wait_time']:.1f} seconds")
    print(f"  Vehicles in System: {agg['avg_vehicles']:.0f} (max: {agg['max_vehicles']})")
    
    # Print per-vehicle-type statistics
    if 'vehicle_types' in agg and agg['vehicle_types']:
        print(f"\nPer-Vehicle-Type Statistics:")
        print(f"{'-'*70}")
        # Show all detected vehicle types dynamically
        for vtype in sorted(agg['vehicle_types'].keys()):
            stats = agg['vehicle_types'][vtype]
            print(f"{vtype.upper():12} | "
                  f"Avg Wait: {stats['avg_wait_time']:6.1f}s | "
                  f"Min Wait: {stats['min_wait_time']:6.1f}s | "
                  f"Max Wait: {stats['max_wait_time']:6.1f}s")
    
    print("="*70)
    print("\nBaseline metrics provide comparison point for RL agent training.")
    print("Use these values to evaluate RL agent improvements.\n")


if __name__ == "__main__":
    # Run baseline evaluation
    metrics = run_baseline(
        num_episodes=3,      # 3 episodes for baseline
        max_steps=500,       # 500 steps per episode
        use_gui=False        # Run headless
    )
    
    # Save metrics
    save_baseline_metrics(metrics)
    
    # Print summary
    print_baseline_summary(metrics)
