import os
import sys
import argparse
import numpy as np
import traci
from stable_baselines3 import PPO
from SumoEnv import SumoEnv, normalize_vehicle_type

# --- Parse Command Line Arguments ---
parser = argparse.ArgumentParser(description="Evaluate PPO agent on different SUMO configs with observation normalization")
parser.add_argument("--config", type=str, default="SUMO_Trinity_Traffic_sim/osm.sumocfg", 
                    help="Path to SUMO config file")
parser.add_argument("--model", type=str, default="models/ppo_mg_road/best_model", 
                    help="Path to trained PPO model")
parser.add_argument("--episodes", type=int, default=5, 
                    help="Number of evaluation episodes")
args = parser.parse_args()

# --- Configuration ---
MODEL_PATH = args.model
NUM_EPISODES = args.episodes
USE_GUI = True
DETERMINISTIC = True
SUMO_CONFIG = args.config
EXPECTED_OBS_SIZE = 43  # Model trained on 14 lanes (3*14+1=43)

# All vehicle types to track
VEHICLE_TYPES = ["car", "bus", "emergency", "auto", "motorcycle", "truck"]

# --- Load Trained Agent ---
print(f"Loading trained PPO model from {MODEL_PATH}...")
try:
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load the model from {MODEL_PATH}.")
    print(e)
    sys.exit("Exiting due to model loading error.")

# --- Observation Normalization Function ---
def normalize_observation(obs, expected_size=EXPECTED_OBS_SIZE):
    """
    Normalize observation to match expected model input size.
    If obs is larger: take first expected_size elements
    If obs is smaller: pad with zeros
    """
    obs = np.array(obs, dtype=np.float32).flatten()
    
    if len(obs) == expected_size:
        return obs
    elif len(obs) > expected_size:
        # Take first expected_size elements (truncate extra lanes)
        print(f"  WARNING: Observation size {len(obs)} > expected {expected_size}. Truncating to match model input.")
        return obs[:expected_size]
    else:
        # Pad with zeros
        print(f"  WARNING: Observation size {len(obs)} < expected {expected_size}. Padding with zeros.")
        padded = np.zeros(expected_size, dtype=np.float32)
        padded[:len(obs)] = obs
        return padded

# --- Environment Setup ---
print("Creating SUMO environment for evaluation...")
# Auto-detect begin time: if using SUMO_Indiranagar_Traffic_sim, start at 0; otherwise use default
begin_time = 0.0 if "SUMO_Indiranagar_Traffic_sim" in SUMO_CONFIG else 28800.0
print(f"DEBUG: Using begin_time={begin_time} for config {SUMO_CONFIG}")
eval_env = SumoEnv(use_gui=USE_GUI, sumocfg_file=SUMO_CONFIG, begin_time=begin_time)

# --- Evaluation Function ---
def run_evaluation(env, agent=None, num_episodes=NUM_EPISODES, agent_name="Agent"):
    """
    Run evaluation with optional agent control or baseline fixed-time control.
    agent=None means use baseline (fixed-time) control
    """
    # Initialize tracking for all vehicle types
    all_episode_travel_times = {}
    episode_results = []
    
    print(f"\nStarting evaluation using {agent_name} for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} / {num_episodes} ({agent_name}) ---")
        
        episode_travel_times = {v_type: [] for v_type in VEHICLE_TYPES}
        depart_times = {}
        veh_types = {}
        veh_source = {}
        action_counts = {'keep': 0, 'change': 0}
        
        depart_traci_count = 0
        depart_inferred_count = 0
        depart_class_mapped = 0
        arrival_without_depart = 0
        
        try:
            obs, info = env.reset()
            obs = normalize_observation(obs)  # Normalize observation
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
        except Exception as e:
            print(f"ERROR: Episode {episode+1}: Failed during reset: {e}")
            continue
        
        current_sim_time = 0.0
        
        try:
            while not terminated and not truncated:
                # Get action from agent or use baseline (no action)
                if agent is not None:
                    action, _states = agent.predict(obs, deterministic=DETERMINISTIC)
                else:
                    # Baseline: always keep the light (action=0)
                    action = 0
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                obs = normalize_observation(obs)  # Normalize observation after each step
                
                # Track action
                action_counts['keep' if action == 0 else 'change'] += 1
                
                # Collect metrics
                if env.traci_conn is not None:
                    try:
                        current_sim_time = env.traci_conn.simulation.getTime()
                        
                        # Process departures
                        departed = []
                        try:
                            departed = env.pop_departed()
                        except Exception:
                            try:
                                departed = env.traci_conn.simulation.getDepartedIDList()
                            except Exception:
                                departed = []
                        
                        for veh_id in departed:
                            try:
                                depart_times[veh_id] = current_sim_time
                                veh_types[veh_id] = env.traci_conn.vehicle.getTypeID(veh_id)
                                depart_traci_count += 1
                            except Exception:
                                pass
                        
                        # Process arrivals
                        arrived = []
                        try:
                            arrived = env.pop_arrived()
                        except Exception:
                            try:
                                arrived = env.traci_conn.simulation.getArrivedIDList()
                            except Exception:
                                arrived = []
                        
                        for veh_id in arrived:
                            try:
                                arrival_time = current_sim_time
                                
                                # Determine depart time
                                if veh_id in depart_times:
                                    depart = depart_times[veh_id]
                                else:
                                    try:
                                        depart_idx = veh_id.rfind('_')
                                        depart_time_from_id = int(veh_id[depart_idx+1:])
                                        depart = depart_time_from_id
                                        depart_inferred_count += 1
                                    except Exception:
                                        depart = current_sim_time
                                        arrival_without_depart += 1
                                
                                # Get vehicle type
                                if veh_id in veh_types:
                                    vtype = veh_types[veh_id]
                                else:
                                    try:
                                        vtype = env.traci_conn.vehicle.getTypeID(veh_id)
                                        depart_class_mapped += 1
                                    except Exception:
                                        continue
                                
                                # Normalize vehicle type
                                vtype = normalize_vehicle_type(vtype)
                                
                                # Calculate travel time
                                travel_time = arrival_time - depart
                                if travel_time > 0:
                                    episode_travel_times[vtype].append(travel_time)
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                steps += 1
                total_reward += reward
        
        except Exception as e:
            print(f"ERROR: Episode {episode+1}: Unexpected error during evaluation loop: {e}")
        
        # Episode summary
        print(f"Episode {episode + 1} finished after {steps} agent steps (Sim Time: {current_sim_time:.2f}). Total Reward: {total_reward:.2f}")
        
        for v_type in VEHICLE_TYPES:
            if episode_travel_times[v_type]:
                avg_time = sum(episode_travel_times[v_type]) / len(episode_travel_times[v_type])
                print(f"  Avg {v_type.capitalize()} Time: {avg_time:.2f}s ({len(episode_travel_times[v_type])} finished)")
        
        print(f"\nEpisode diagnostics:")
        print(f"  Depart types obtained from Traci: {depart_traci_count}")
        print(f"  Depart types inferred from id: {depart_inferred_count}")
        print(f"  Depart types mapped from vehicle class: {depart_class_mapped}")
        print(f"  Arrivals without recorded depart time: {arrival_without_depart}")
        print(f"  Agent actions: Keep={action_counts['keep']}, Change={action_counts['change']}")
        
        # Store episode results
        ep_res = {'episode': episode + 1}
        for v_type in VEHICLE_TYPES:
            if episode_travel_times[v_type]:
                ep_res[f'{v_type}_count'] = len(episode_travel_times[v_type])
                ep_res[f'{v_type}_avg'] = float(sum(episode_travel_times[v_type]) / len(episode_travel_times[v_type]))
            else:
                ep_res[f'{v_type}_count'] = 0
                ep_res[f'{v_type}_avg'] = 0.0
        
        all_episode_travel_times[episode + 1] = episode_travel_times
        episode_results.append(ep_res)
    
    return episode_results, all_episode_travel_times

# --- Run Baseline Evaluation ---
print(f"\n{'='*110}")
print("BASELINE EVALUATION (Fixed-Time Signal Control)")
print(f"{'='*110}")
baseline_results, baseline_times = run_evaluation(eval_env, agent=None, num_episodes=NUM_EPISODES, agent_name="Baseline")

# --- Run PPO Agent Evaluation ---
print(f"\n{'='*110}")
print("PPO AGENT EVALUATION (Learned Control)")
print(f"{'='*110}")
ppo_results, ppo_times = run_evaluation(eval_env, agent=model, num_episodes=NUM_EPISODES, agent_name="PPO Agent")

# --- Comparison ---
print(f"\n{'='*110}")
print("OVERALL COMPARISON: BASELINE vs PPO AGENT")
print(f"{'='*110}")

for v_type in VEHICLE_TYPES:
    baseline_list = baseline_times.get(v_type, [])
    ppo_list = ppo_times.get(v_type, [])
    
    if baseline_list and ppo_list:
        baseline_avg = sum(baseline_list) / len(baseline_list)
        ppo_avg = sum(ppo_list) / len(ppo_list)
        improvement = ((baseline_avg - ppo_avg) / baseline_avg) * 100
        direction = "Faster" if improvement > 0 else "Slower"
        print(f"  {v_type.capitalize()}: {abs(improvement):.1f}% {direction} ({baseline_avg:.2f}s -> {ppo_avg:.2f}s)")
    else:
        if not baseline_list and not ppo_list:
            print(f"  {v_type.capitalize()}: No data available")
        else:
            print(f"  {v_type.capitalize()}: Incomplete data (missing from one system)")

# --- Detailed Comparison Tables (All Vehicle Types) ---
print(f"\n{'='*160}")
print("BASELINE EVALUATION RESULTS (Fixed-Time Signal) - ALL VEHICLE TYPES")
print(f"{'='*160}")
print(f"{'Ep':<4} {'Cars':<8} {'Car(s)':<10} {'Bus':<8} {'Bus(s)':<10} {'Emerg':<8} {'Emer(s)':<10} {'Auto':<8} {'Auto(s)':<10} {'Moto':<8} {'Moto(s)':<10} {'Truck':<8} {'Trk(s)':<10}")
print("-"*160)
for ep_res in baseline_results:
    print(f"{ep_res['episode']:<4} {ep_res.get('car_count', 0):<8} {ep_res.get('car_avg', 0):<10.2f} {ep_res.get('bus_count', 0):<8} {ep_res.get('bus_avg', 0):<10.2f} {ep_res.get('emergency_count', 0):<8} {ep_res.get('emergency_avg', 0):<10.2f} {ep_res.get('auto_count', 0):<8} {ep_res.get('auto_avg', 0):<10.2f} {ep_res.get('motorcycle_count', 0):<8} {ep_res.get('motorcycle_avg', 0):<10.2f} {ep_res.get('truck_count', 0):<8} {ep_res.get('truck_avg', 0):<10.2f}")
print(f"{'='*160}")

print(f"\n{'='*160}")
print("PPO AGENT EVALUATION RESULTS (Learned Control) - ALL VEHICLE TYPES")
print(f"{'='*160}")
print(f"{'Ep':<4} {'Cars':<8} {'Car(s)':<10} {'Bus':<8} {'Bus(s)':<10} {'Emerg':<8} {'Emer(s)':<10} {'Auto':<8} {'Auto(s)':<10} {'Moto':<8} {'Moto(s)':<10} {'Truck':<8} {'Trk(s)':<10}")
print("-"*160)
for ep_res in ppo_results:
    print(f"{ep_res['episode']:<4} {ep_res.get('car_count', 0):<8} {ep_res.get('car_avg', 0):<10.2f} {ep_res.get('bus_count', 0):<8} {ep_res.get('bus_avg', 0):<10.2f} {ep_res.get('emergency_count', 0):<8} {ep_res.get('emergency_avg', 0):<10.2f} {ep_res.get('auto_count', 0):<8} {ep_res.get('auto_avg', 0):<10.2f} {ep_res.get('motorcycle_count', 0):<8} {ep_res.get('motorcycle_avg', 0):<10.2f} {ep_res.get('truck_count', 0):<8} {ep_res.get('truck_avg', 0):<10.2f}")
print(f"{'='*160}")

# --- Close Environment ---
print("\nEvaluation script completed successfully.")
