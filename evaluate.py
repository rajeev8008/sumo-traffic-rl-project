import os
import sys
import numpy as np
import traci
from stable_baselines3 import PPO
from SumoEnv import SumoEnv

# --- Configuration ---
MODEL_PATH = "models/ppo_mg_road/best_model"
NUM_EPISODES = 5
USE_GUI = False
DETERMINISTIC = True

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
eval_env = SumoEnv(use_gui=USE_GUI, sumocfg_file="SUMO_Trinity_Traffic_sim/osm.sumocfg")

# --- Evaluation Loop ---
all_episode_travel_times = {"car": [], "bus": [], "emergency": []}
episode_results = []

print(f"\nStarting evaluation using the agent for {NUM_EPISODES} episodes...")

for episode in range(NUM_EPISODES):
    print(f"\n--- Episode {episode + 1} / {NUM_EPISODES} ---")
    
    episode_travel_times = {"car": [], "bus": [], "emergency": []}
    depart_times = {}
    veh_types = {}
    veh_source = {}  # ADDED - Initialize this variable
    action_counts = {'keep': 0, 'change': 0}
    
    # ADDED - Initialize diagnostics counters
    depart_traci_count = 0
    depart_inferred_count = 0
    depart_class_mapped = 0
    arrival_without_depart = 0
    
    try:
        obs, info = eval_env.reset()
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
            # Get action from agent
            action, _states = model.predict(obs, deterministic=DETERMINISTIC)
            
            # Step environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Track action
            action_counts['keep' if action == 0 else 'change'] += 1
            
            # Collect metrics
            if eval_env.traci_conn is not None:
                try:
                    current_sim_time = eval_env.traci_conn.simulation.getTime()
                    
                    # Process departures
                    departed = []
                    try:
                        departed = eval_env.pop_departed()
                    except Exception:
                        try:
                            departed = eval_env.traci_conn.simulation.getDepartedIDList()
                        except Exception:
                            departed = []
                    
                    for veh_id in departed:
                        type_id = None
                        source = None
                        
                        try:
                            type_id = eval_env.traci_conn.vehicle.getTypeID(veh_id)
                            source = 'traci'
                        except:
                            type_id = None
                            source = None
                        
                        # If TraCI didn't return a useful type, try to infer from vehicle id
                        if type_id is None:
                            try:
                                base = veh_id.split('.')[0]
                                if "car" in base:
                                    type_id = "car"
                                    source = 'inferred'
                                elif "bus" in base:
                                    type_id = "bus"
                                    source = 'inferred'
                                elif "emergency" in base:
                                    type_id = "emergency"
                                    source = 'inferred'
                                else:
                                    try:
                                        vclass = eval_env.traci_conn.vehicle.getVehicleClass(veh_id)
                                        if vclass == "passenger":
                                            type_id = "car"
                                            source = 'class'
                                        elif vclass == "emergency":
                                            type_id = "emergency"
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
                        arrived_vehicle_ids = eval_env.pop_arrived()
                    except Exception:
                        try:
                            arrived_vehicle_ids = eval_env.traci_conn.simulation.getArrivedIDList()
                        except Exception:
                            arrived_vehicle_ids = []
                    
                    for veh_id in arrived_vehicle_ids:
                        try:
                            vtype = veh_types.get(veh_id)
                            dep_time = depart_times.get(veh_id)
                            
                            # If we don't have a recorded type, try to infer
                            if vtype is None:
                                try:
                                    base = veh_id.split('.')[0]
                                    if "car" in base:
                                        vtype = "car"
                                    elif "bus" in base:
                                        vtype = "bus"
                                    elif "emergency" in base:
                                        vtype = "emergency"
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
        'cars_avg': 0.0, 'bus_avg': 0.0, 'emerg_avg': 0.0
    }
    
    for v_type in ["car", "bus", "emergency"]:
        times = episode_travel_times.get(v_type, [])
        if times:
            avg_time = sum(times) / len(times)
            print(f"  Avg {v_type.capitalize()} Time: {avg_time:.2f}s ({len(times)} finished)")
            if v_type == "car":
                ep_result['cars_count'], ep_result['cars_avg'] = len(times), avg_time
            elif v_type == "bus":
                ep_result['bus_count'], ep_result['bus_avg'] = len(times), avg_time
            else:
                ep_result['emerg_count'], ep_result['emerg_avg'] = len(times), avg_time
        else:
            print(f"  No {v_type}s finished.")
    
    print("\nEpisode diagnostics:")
    print(f"  Depart types obtained from Traci: {depart_traci_count}")
    print(f"  Depart types inferred from id: {depart_inferred_count}")
    print(f"  Depart types mapped from vehicle class: {depart_class_mapped}")
    print(f"  Arrivals without recorded depart time: {arrival_without_depart}")
    print(f"  Agent actions: Keep={action_counts['keep']}, Change={action_counts['change']}")
    
    episode_results.append(ep_result)

# --- Final Summary ---
print(f"\n--- Overall Evaluation Results (Agent Performance) ---")

for v_type in ["car", "bus", "emergency"]:
    times = all_episode_travel_times.get(v_type, [])
    if times:
        avg = sum(times) / len(times)
        print(f"Average {v_type.capitalize()} Travel Time:   {avg:.2f} seconds ({len(times)} finished across {NUM_EPISODES} episodes)")
    else:
        print(f"No {v_type} vehicles finished across all episodes.")

eval_env.close()
print("\nEvaluation script finished.")

# --- Summary Table ---
print(f"\n{'='*110}")
print("PER-EPISODE SUMMARY TABLE")
print(f"{'='*110}")
print(f"{'Episode':<10} {'Cars Fin.':<12} {'Avg Car (s)':<15} {'Buses Fin.':<12} {'Avg Bus (s)':<15} {'Emerg Fin.':<12} {'Avg Emerg (s)':<15}")
print("-"*110)
for ep_res in episode_results:
    print(f"{ep_res['episode']:<10} {ep_res['cars_count']:<12} {ep_res['cars_avg']:<15.2f} {ep_res['bus_count']:<12} {ep_res['bus_avg']:<15.2f} {ep_res['emerg_count']:<12} {ep_res['emerg_avg']:<15.2f}")
print(f"{'='*110}")