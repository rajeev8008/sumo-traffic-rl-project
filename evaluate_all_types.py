import os
import sys
import numpy as np
import traci
from stable_baselines3 import PPO
from SumoEnv import SumoEnv, normalize_vehicle_type

# --- Configuration ---
MODEL_PATH = "models/ppo_mg_road/best_model"
NUM_EPISODES = 5
USE_GUI = False
DETERMINISTIC = True
SUMO_CONFIG = "SUMO_Trinity_Traffic_sim/osm.sumocfg"

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

# --- Environment Setup ---
print("Creating SUMO environment for evaluation...")
eval_env = SumoEnv(use_gui=USE_GUI, sumocfg_file=SUMO_CONFIG)

# --- Evaluation Function ---
def run_evaluation(env, agent=None, num_episodes=NUM_EPISODES, agent_name="Agent"):
    """
    Run evaluation with optional agent control or baseline fixed-time control.
    agent=None means use baseline (fixed-time) control
    """
    # Initialize tracking for all vehicle types
    all_episode_travel_times = {v_type: [] for v_type in VEHICLE_TYPES}
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
                            type_id = None
                            source = None
                            
                            try:
                                type_id = env.traci_conn.vehicle.getTypeID(veh_id)
                                source = 'traci'
                                type_id = normalize_vehicle_type(type_id)
                            except:
                                type_id = None
                                source = None
                            
                            # If TraCI didn't return a useful type, try to infer from vehicle id
                            if type_id is None:
                                try:
                                    base = veh_id.split('.')[0].lower()
                                    # Check for each type
                                    if any(x in base for x in ["ambulance", "fire_truck", "emergency"]):
                                        type_id = "emergency"
                                        source = 'inferred'
                                    elif "bus" in base:
                                        type_id = "bus"
                                        source = 'inferred'
                                    elif "car" in base or "default_car" in base:
                                        type_id = "car"
                                        source = 'inferred'
                                    elif "motorcycle" in base or "moto" in base:
                                        type_id = "motorcycle"
                                        source = 'inferred'
                                    elif "truck" in base:
                                        type_id = "truck"
                                        source = 'inferred'
                                    elif "auto" in base or "taxi" in base:
                                        type_id = "auto"
                                        source = 'inferred'
                                    else:
                                        try:
                                            vclass = env.traci_conn.vehicle.getVehicleClass(veh_id)
                                            type_id = normalize_vehicle_type(vclass)
                                            source = 'class'
                                        except:
                                            type_id = None
                                            source = None
                                except:
                                    type_id = None
                                    source = None
                            
                            depart_times[veh_id] = current_sim_time
                            veh_types[veh_id] = type_id
                            veh_source[veh_id] = source
                            
                            if source == 'traci':
                                depart_traci_count += 1
                            elif source == 'inferred':
                                depart_inferred_count += 1
                            elif source == 'class':
                                depart_class_mapped += 1
                        
                        # Process arrivals
                        arrived_vehicle_ids = []
                        try:
                            arrived_vehicle_ids = env.pop_arrived()
                        except Exception:
                            try:
                                arrived_vehicle_ids = env.traci_conn.simulation.getArrivedIDList()
                            except Exception:
                                arrived_vehicle_ids = []
                        
                        for veh_id in arrived_vehicle_ids:
                            try:
                                vtype = veh_types.get(veh_id)
                                dep_time = depart_times.get(veh_id)
                                
                                # If we don't have a recorded type, try to infer
                                if vtype is None:
                                    try:
                                        base = veh_id.split('.')[0].lower()
                                        if any(x in base for x in ["ambulance", "fire_truck", "emergency"]):
                                            vtype = "emergency"
                                        elif "bus" in base:
                                            vtype = "bus"
                                        elif "car" in base or "default_car" in base:
                                            vtype = "car"
                                        elif "motorcycle" in base or "moto" in base:
                                            vtype = "motorcycle"
                                        elif "truck" in base:
                                            vtype = "truck"
                                        elif "auto" in base or "taxi" in base:
                                            vtype = "auto"
                                        
                                        if vtype is not None:
                                            veh_types[veh_id] = vtype
                                            veh_source[veh_id] = 'inferred'
                                    except:
                                        vtype = None
                                
                                # Record travel time
                                if vtype in episode_travel_times and dep_time is not None:
                                    duration = current_sim_time - dep_time
                                    episode_travel_times[vtype].append(duration)
                                    all_episode_travel_times[vtype].append(duration)
                                else:
                                    if dep_time is None:
                                        arrival_without_depart += 1
                                
                                # Cleanup
                                depart_times.pop(veh_id, None)
                                veh_types.pop(veh_id, None)
                                veh_source.pop(veh_id, None)
                            except Exception as e_inner:
                                print(f"WARNING: Episode {episode+1}: Error processing arrived {veh_id}: {e_inner}")
                    
                    except Exception as e_metric:
                        print(f"WARNING: Episode {episode+1}: Unexpected error during metric collection: {e_metric}. Skipping metrics for this step.")
                
                total_reward += reward
                steps += 1
        
        except Exception as e:
            print(f"ERROR: Episode {episode+1}: Unexpected error during evaluation loop: {e}")
        
        # Print episode summary
        print(f"Episode {episode + 1} finished after {steps} agent steps (Sim Time: {current_sim_time:.2f}). Total Reward: {total_reward:.2f}")
        
        ep_result = {
            'episode': episode + 1,
            'cars_count': 0, 'bus_count': 0, 'emerg_count': 0,
            'auto_count': 0, 'motorcycle_count': 0, 'truck_count': 0,
            'cars_avg': 0.0, 'bus_avg': 0.0, 'emerg_avg': 0.0,
            'auto_avg': 0.0, 'motorcycle_avg': 0.0, 'truck_avg': 0.0
        }
        
        for v_type in VEHICLE_TYPES:
            times = episode_travel_times.get(v_type, [])
            if times:
                avg_time = sum(times) / len(times)
                print(f"  Avg {v_type.capitalize()} Time: {avg_time:.2f}s ({len(times)} finished)")
                if v_type == "car":
                    ep_result['cars_count'], ep_result['cars_avg'] = len(times), avg_time
                elif v_type == "bus":
                    ep_result['bus_count'], ep_result['bus_avg'] = len(times), avg_time
                elif v_type == "emergency":
                    ep_result['emerg_count'], ep_result['emerg_avg'] = len(times), avg_time
                elif v_type == "auto":
                    ep_result['auto_count'], ep_result['auto_avg'] = len(times), avg_time
                elif v_type == "motorcycle":
                    ep_result['motorcycle_count'], ep_result['motorcycle_avg'] = len(times), avg_time
                elif v_type == "truck":
                    ep_result['truck_count'], ep_result['truck_avg'] = len(times), avg_time
            else:
                print(f"  No {v_type}s finished.")
        
        print("\nEpisode diagnostics:")
        print(f"  Depart types obtained from Traci: {depart_traci_count}")
        print(f"  Depart types inferred from id: {depart_inferred_count}")
        print(f"  Depart types mapped from vehicle class: {depart_class_mapped}")
        print(f"  Arrivals without recorded depart time: {arrival_without_depart}")
        print(f"  Agent actions: Keep={action_counts['keep']}, Change={action_counts['change']}")
        
        episode_results.append(ep_result)
    
    return all_episode_travel_times, episode_results

# --- Run Baseline Evaluation ---
print("\n" + "="*110)
print("BASELINE EVALUATION (Fixed-Time Signal Control)")
print("="*110)
baseline_times, baseline_results = run_evaluation(eval_env, agent=None, num_episodes=NUM_EPISODES, agent_name="Baseline")

# --- Run PPO Agent Evaluation ---
print("\n" + "="*110)
print("PPO AGENT EVALUATION (Learned Control)")
print("="*110)
ppo_times, ppo_results = run_evaluation(eval_env, agent=model, num_episodes=NUM_EPISODES, agent_name="PPO Agent")

eval_env.close()

# --- Final Summary ---
print(f"\n{'='*110}")
print("OVERALL COMPARISON: BASELINE vs PPO AGENT")
print(f"{'='*110}\n")

# Compare overall metrics
def print_summary(agent_name, times_dict):
    print(f"{agent_name}:")
    for v_type in VEHICLE_TYPES:
        times = times_dict.get(v_type, [])
        if times:
            avg = sum(times) / len(times)
            print(f"  Average {v_type.capitalize()} Travel Time:   {avg:.2f} seconds ({len(times)} finished across {NUM_EPISODES} episodes)")
        else:
            print(f"  No {v_type} vehicles finished across all episodes.")
    print()

print_summary("BASELINE (Fixed-Time Control)", baseline_times)
print_summary("PPO AGENT (Learned Control)", ppo_times)

# Print improvement percentages
print("IMPROVEMENT (PPO vs Baseline):")
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
    print(f"{ep_res['episode']:<4} {ep_res['cars_count']:<8} {ep_res['cars_avg']:<10.2f} {ep_res['bus_count']:<8} {ep_res['bus_avg']:<10.2f} {ep_res['emerg_count']:<8} {ep_res['emerg_avg']:<10.2f} {ep_res['auto_count']:<8} {ep_res['auto_avg']:<10.2f} {ep_res['motorcycle_count']:<8} {ep_res['motorcycle_avg']:<10.2f} {ep_res['truck_count']:<8} {ep_res['truck_avg']:<10.2f}")
print(f"{'='*160}")

print(f"\n{'='*160}")
print("PPO AGENT EVALUATION RESULTS (Learned Control) - ALL VEHICLE TYPES")
print(f"{'='*160}")
print(f"{'Ep':<4} {'Cars':<8} {'Car(s)':<10} {'Bus':<8} {'Bus(s)':<10} {'Emerg':<8} {'Emer(s)':<10} {'Auto':<8} {'Auto(s)':<10} {'Moto':<8} {'Moto(s)':<10} {'Truck':<8} {'Trk(s)':<10}")
print("-"*160)
for ep_res in ppo_results:
    print(f"{ep_res['episode']:<4} {ep_res['cars_count']:<8} {ep_res['cars_avg']:<10.2f} {ep_res['bus_count']:<8} {ep_res['bus_avg']:<10.2f} {ep_res['emerg_count']:<8} {ep_res['emerg_avg']:<10.2f} {ep_res['auto_count']:<8} {ep_res['auto_avg']:<10.2f} {ep_res['motorcycle_count']:<8} {ep_res['motorcycle_avg']:<10.2f} {ep_res['truck_count']:<8} {ep_res['truck_avg']:<10.2f}")
print(f"{'='*160}")

# --- Close Environment ---
print("\nEvaluation script completed successfully.")
