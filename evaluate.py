import os
import sys
import numpy as np
import traci # Keep traci import for specific calls if needed outside env
from stable_baselines3 import PPO

# Import your custom environment
from SumoEnv import SumoEnv # Assuming SumoEnv.py is in the same directory

# --- Configuration ---
MODEL_PATH = "models/ppo_sumo_model/ppo_sumo_final_model.zip" # Path to your saved model
NUM_EPISODES = 5 # Number of episodes to run for evaluation
# MAX_STEPS_PER_EPISODE is now handled by the environment's internal limit
USE_GUI = False # Set to False for automated evaluation (no GUI)

# --- Load Trained Agent ---
print(f"Loading trained PPO model from {MODEL_PATH}...")
try:
    # No need to pass env here, policy/spaces are loaded from the zip
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load the model from {MODEL_PATH}.")
    print(e)
    sys.exit("Exiting due to model loading error.")

# --- Environment Setup for Evaluation ---
# Create the environment instance. SB3 interacts via this instance.
print("Creating SUMO environment for evaluation...")
# Pass the USE_GUI flag to the environment constructor
eval_env = SumoEnv(use_gui=USE_GUI, sumocfg_file="map.sumocfg2")

# --- Evaluation Loop ---
# Initialize with CORRECT keys matching vType IDs
all_episode_travel_times = { "car": [], "bus": [], "emergency": [] }
# Track per-episode results for final summary table
episode_results = []  # List of dicts: {episode: X, cars: N, bus: N, emergency: N, avg_car: float, avg_bus: float, avg_emerg: float}

print(f"\nStarting evaluation using the agent for {NUM_EPISODES} episodes...")

for episode in range(NUM_EPISODES):
    print(f"\n--- Episode {episode + 1} / {NUM_EPISODES} ---")

    # --- Data Collection for this episode ---
    # Initialize with CORRECT keys matching vType IDs
    episode_travel_times = { "car": [], "bus": [], "emergency": [] }
    depart_times = {}   # veh_id -> depart simulation time
    veh_types = {}      # veh_id -> vehicle type ('car', 'bus', 'emergency')
    veh_source = {}     # veh_id -> how we obtained the type: 'traci'|'inferred'|'class'|None
    # Diagnostics counters
    depart_traci_count = 0
    depart_inferred_count = 0
    depart_class_mapped = 0
    arrival_without_depart = 0
    
    # Track agent actions taken in this episode
    action_counts = {'keep': 0, 'change': 0}

    # --- Reset Environment (Starts SUMO via SumoEnv's reset) ---
    try:
        obs, info = eval_env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0 # Agent steps count
    except Exception as e_reset:
        print(f"ERROR: Episode {episode+1}: Failed during environment reset: {e_reset}")
        continue # Skip to the next episode

    current_sim_time = 0.0 # Track current simulation time

    try:
        # Loop until the episode ends (terminated or truncated)
        while not terminated and not truncated:
            # --- Get Action from Agent ---
            # Use deterministic=False for evaluation (agent explores with stochastic policy)
            action, _states = model.predict(obs, deterministic=False)

            # --- Step Environment with Agent's Action ---
            # env.step applies the action, runs SUMO, gets new obs/reward
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Track action taken
            if action == 0:
                action_counts['keep'] += 1
            else:
                action_counts['change'] += 1

            # --- Metric Collection (Requires access to TraCI instance within env) ---
            # Get current simulation time *after* the step
            if eval_env.traci_conn is not None:
                try:
                    current_sim_time = eval_env.traci_conn.simulation.getTime()

                    # Record Departures (from env buffer populated during env.step())
                    departed = []
                    try:
                        departed = eval_env.pop_departed()
                    except Exception:
                        # Fallback to direct TraCI call if env doesn't supply buffer
                        try:
                            departed = eval_env.traci_conn.simulation.getDepartedIDList()
                        except Exception:
                            departed = []
                    for veh_id in departed:
                        type_id = None
                        source = None
                        # Primarily use getTypeID which should match 'car', 'bus', 'emergency'
                        try:
                            type_id = eval_env.traci_conn.vehicle.getTypeID(veh_id)
                            source = 'traci'
                        except traci.TraCIException as e_type:
                            # If getTypeID fails, we'll try to infer below
                            print(f"WARNING: Could not get vTypeID for departing {veh_id}: {e_type}. Will attempt to infer from ID.")
                            type_id = None
                            source = None
                        except Exception as e_gen:
                            print(f"ERROR: Unexpected error getting vTypeID for departing {veh_id}: {e_gen}. Will attempt to infer from ID.")
                            type_id = None
                            source = None

                        # If TraCI didn't return a useful type, try to infer from the vehicle id
                        # (common when flows generate ids like "car_NS.0", "bus_SN.0").
                        if type_id is None:
                            try:
                                base = veh_id.split('.')[0]  # e.g. "car_NS"
                                # Simple heuristics to map id -> type key
                                if base.startswith("car") or "car" in base:
                                    type_id = "car"
                                    source = 'inferred'
                                elif base.startswith("bus") or "bus" in base:
                                    type_id = "bus"
                                    source = 'inferred'
                                elif base.startswith("emergency") or "emergency" in base:
                                    type_id = "emergency"
                                    source = 'inferred'
                                else:
                                    # As a last resort, try vehicle class and map 'passenger' -> 'car'
                                    try:
                                        vclass = eval_env.traci_conn.vehicle.getVehicleClass(veh_id)
                                        if vclass == "passenger":
                                            type_id = "car"
                                            source = 'class'
                                        elif vclass == "emergency":
                                            type_id = "emergency"
                                            source = 'class'
                                        else:
                                            type_id = None
                                            source = None
                                    except Exception:
                                        type_id = None
                                        source = None
                            except Exception:
                                type_id = None
                                source = None

                        depart_times[veh_id] = current_sim_time # Store sim time
                        # Store the type_id (may be None if we couldn't infer)
                        veh_types[veh_id] = type_id
                        veh_source[veh_id] = source
                        # bump counters
                        if source == 'traci':
                            depart_traci_count += 1
                        elif source == 'inferred':
                            depart_inferred_count += 1
                        elif source == 'class':
                            depart_class_mapped += 1

                    # Process Arrivals (from env buffer populated during env.step())
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
                            vtype = veh_types.get(veh_id) # May be None if we couldn't record on depart
                            dep_time = depart_times.get(veh_id)

                            # --- DEBUG PRINT ADDED HERE ---
                            print(f"DEBUG: Processing arrival for {veh_id}. Recorded type: '{vtype}'. Checking against keys: {list(episode_travel_times.keys())}")
                            # --- END DEBUG PRINT ---

                            # If we don't have a recorded type, try to infer it from the vehicle id
                            # Example ids: 'car_NS.0', 'bus_SN.1', 'emergency_EW.0'
                            if vtype is None:
                                try:
                                    base = veh_id.split('.')[0]
                                    if base.startswith("car") or "car" in base:
                                        vtype = "car"
                                    elif base.startswith("bus") or "bus" in base:
                                        vtype = "bus"
                                    elif base.startswith("emergency") or "emergency" in base:
                                        vtype = "emergency"
                                    # Save inferred type for completeness
                                    if vtype is not None:
                                        veh_types[veh_id] = vtype
                                        veh_source[veh_id] = 'inferred'
                                        print(f"DEBUG: Inferred type '{vtype}' for {veh_id} from id.")
                                except Exception:
                                    vtype = None

                            # Check against the CORRECT keys
                            if vtype in episode_travel_times and dep_time is not None:
                                duration = current_sim_time - dep_time
                                episode_travel_times[vtype].append(duration)
                                all_episode_travel_times[vtype].append(duration) # Add to overall list
                                print(f"DEBUG: ---> Recorded duration {duration:.2f} for {vtype} {veh_id}") # Confirm recording
                            else:
                                # If depart time missing, count for diagnostics
                                if dep_time is None:
                                    arrival_without_depart += 1


                            # Clean up maps
                            if veh_id in depart_times: del depart_times[veh_id]
                            if veh_id in veh_types: del veh_types[veh_id]
                            if veh_id in veh_source: del veh_source[veh_id]
                        except Exception as e_inner:
                            print(f"WARNING: Episode {episode+1}: Error processing arrived {veh_id}: {e_inner}")
                except traci.TraCIException as e_traci_metric:
                    print(f"WARNING: Episode {episode+1}: TraCI error during metric collection: {e_traci_metric}. Skipping metrics for this step.")
                except Exception as e_metric:
                    print(f"WARNING: Episode {episode+1}: Unexpected error during metric collection: {e_metric}. Skipping metrics for this step.")

            else:
                 print("WARNING: Traci connection lost, cannot collect metrics for this step.")

            total_reward += reward
            steps += 1 # Count agent steps

    except Exception as e:
        print(f"ERROR: Episode {episode+1}: Unexpected error during evaluation loop: {e}")
    finally:
        # env.reset() or env.close() will handle closing SUMO
        pass

    # --- Print Episode Summary ---
    # Use CORRECT keys here too
    print(f"Episode {episode + 1} finished after {steps} agent steps (Sim Time: {current_sim_time:.2f}). Total Reward: {total_reward:.2f}")
    
    # Collect per-episode results
    ep_result = {
        'episode': episode + 1,
        'cars_count': 0,
        'bus_count': 0,
        'emerg_count': 0,
        'cars_avg': 0.0,
        'bus_avg': 0.0,
        'emerg_avg': 0.0
    }
    
    for v_type in ["car", "bus", "emergency"]: # Iterate using correct keys
        times = episode_travel_times.get(v_type, []) # Use .get() for safety
        if times:
            avg_time = sum(times) / len(times)
            print(f"  Avg {v_type.capitalize()} Time: {avg_time:.2f}s ({len(times)} finished)")
            # Store in result dict
            if v_type == "car":
                ep_result['cars_count'] = len(times)
                ep_result['cars_avg'] = avg_time
            elif v_type == "bus":
                ep_result['bus_count'] = len(times)
                ep_result['bus_avg'] = avg_time
            elif v_type == "emergency":
                ep_result['emerg_count'] = len(times)
                ep_result['emerg_avg'] = avg_time
        else:
            print(f"  No {v_type}s finished.")
    
    episode_results.append(ep_result)

    # Episode diagnostics
    print("\nEpisode diagnostics:")
    print(f"  Depart types obtained from Traci: {depart_traci_count}")
    print(f"  Depart types inferred from id: {depart_inferred_count}")
    print(f"  Depart types mapped from vehicle class: {depart_class_mapped}")
    print(f"  Arrivals without recorded depart time: {arrival_without_depart}")
    print(f"  Agent actions: Keep={action_counts['keep']}, Change={action_counts['change']}")


# --- Final Evaluation Summary ---
print("\n--- Overall Evaluation Results (Agent Performance) ---")

# Use CORRECT keys for final calculation and printing
# Calculate Average Car Travel Time
num_cars = len(all_episode_travel_times.get("car", [])) # Use .get()
if num_cars > 0:
    avg_car_time = sum(all_episode_travel_times["car"]) / num_cars
    print(f"Average Car Travel Time:   {avg_car_time:.2f} seconds ({num_cars} finished across {NUM_EPISODES} episodes)")
else:
    print("No cars finished across all episodes.")

# Calculate Average Bus Travel Time
num_buses = len(all_episode_travel_times.get("bus", [])) # Use .get()
if num_buses > 0:
    avg_bus_time = sum(all_episode_travel_times["bus"]) / num_buses
    print(f"Average Bus Travel Time:   {avg_bus_time:.2f} seconds ({num_buses} finished across {NUM_EPISODES} episodes)")
else:
    print("No buses finished across all episodes.")

# Calculate Average Emergency Transit Time
num_emergency = len(all_episode_travel_times.get("emergency", [])) # Use .get()
if num_emergency > 0:
    avg_emergency_time = sum(all_episode_travel_times["emergency"]) / num_emergency
    print(f"Average Emergency Transit Time: {avg_emergency_time:.2f} seconds ({num_emergency} finished across {NUM_EPISODES} episodes)")
else:
    print("No emergency vehicles finished across all episodes.")

# --- Close the environment ---
eval_env.close()
print("\nEvaluation script finished.")

# --- Per-Episode Summary Table ---
print("\n" + "="*110)
print("PER-EPISODE SUMMARY TABLE")
print("="*110)
print(f"{'Episode':<10} {'Cars Fin.':<12} {'Avg Car (s)':<15} {'Buses Fin.':<12} {'Avg Bus (s)':<15} {'Emerg Fin.':<12} {'Avg Emerg (s)':<15}")
print("-"*110)
for ep_res in episode_results:
    ep_num = ep_res['episode']
    cars_cnt = ep_res['cars_count']
    cars_avg = ep_res['cars_avg']
    bus_cnt = ep_res['bus_count']
    bus_avg = ep_res['bus_avg']
    emerg_cnt = ep_res['emerg_count']
    emerg_avg = ep_res['emerg_avg']
    print(f"{ep_num:<10} {cars_cnt:<12} {cars_avg:<15.2f} {bus_cnt:<12} {bus_avg:<15.2f} {emerg_cnt:<12} {emerg_avg:<15.2f}")
print("="*110)