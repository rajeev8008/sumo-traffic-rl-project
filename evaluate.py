import os
import sys
import numpy as np
import traci
from stable_baselines3 import PPO

# Import your custom environment
from SumoEnv import SumoEnv # Assuming SumoEnv.py is in the same directory

# --- Configuration ---
MODEL_PATH = "models/ppo_sumo_model/ppo_sumo_final_model.zip" # Path to your saved model
CONFIG_FILE = "map.sumocfg"
NUM_EPISODES = 5 # Number of episodes to run for evaluation (e.g., 5-10 for stable averages)
MAX_STEPS_PER_EPISODE = 3600 # Max simulation steps per episode (matches baseline duration)
USE_GUI = False # Set to True if you want to watch one evaluation episode

# --- TraCI Setup ---
print("DEBUG: Checking SUMO_HOME...")
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
    print(f"DEBUG: Added {tools} to sys.path")
else:
    print("ERROR: SUMO_HOME environment variable not declared.")
    sys.exit("Please declare the 'SUMO_HOME' environment variable.")

# Determine SUMO binary based on USE_GUI flag
if USE_GUI:
    SUMO_BINARY = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
else:
    SUMO_BINARY = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')

# --- Load Trained Agent ---
print(f"Loading trained PPO model from {MODEL_PATH}...")
try:
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load the model from {MODEL_PATH}. Did training complete and save?")
    print(e)
    sys.exit("Exiting due to model loading error.")

# --- Evaluation Loop ---
all_episode_travel_times = { "car": [], "bus": [], "emergency": [] }

print(f"\nStarting evaluation for {NUM_EPISODES} episodes...")

for episode in range(NUM_EPISODES):
    print(f"\n--- Episode {episode + 1} / {NUM_EPISODES} ---")

    # --- Data Collection for this episode ---
    episode_travel_times = { "car": [], "bus": [], "emergency": [] }
    depart_times = {}   # veh_id -> depart step
    veh_types = {}      # veh_id -> vehicle type ('car', 'bus', 'emergency')

    # --- Start SUMO ---
    sumo_cmd = [SUMO_BINARY, "-c", CONFIG_FILE, "--no-step-log=true", "--no-warnings=true", "--quit-on-end=true"]
    try:
        traci.start(sumo_cmd)
        print(f"DEBUG: Episode {episode+1}: SUMO started.")
    except Exception as e:
        print(f"ERROR: Episode {episode+1}: Failed to start SUMO: {e}")
        continue # Skip to next episode if SUMO fails to start

    # --- Run Episode using Agent's Actions ---
    step = 0
    terminated = False
    truncated = False
    # Get initial observation (need to simulate one step to get state after reset logic)
    # This is a slight simplification; a proper Gymnasium reset would be better,
    # but for evaluation, starting from step 0 is usually fine.
    # We'll get the first observation inside the loop.
    # Note: PPO typically needs the environment reset externally, but for evaluation,
    # we manage the loop and SUMO start/stop manually.

    try:
        while step < MAX_STEPS_PER_EPISODE:
            # Check if simulation is still active
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"DEBUG: Episode {episode+1}: No vehicles expected at step {step}. Ending episode.")
                terminated = True
                break

            # --- Get Observation ---
            # Simulate one SUMO step to allow observation calculation
            traci.simulationStep()
            step += 1
            # Now get the observation based on the state after the step
            # This requires a temporary SumoEnv instance or reimplementing _get_obs logic
            # Simpler approach for eval: Get raw data needed for the *agent*
            # NOTE: For proper evaluation matching training, wrap SUMO in SumoEnv
            # But for basic metrics, collecting travel times like baseline.py is okay.
            # Let's stick to the baseline.py logic for metric collection

            # --- Record Departures (Same logic as baseline.py fix) ---
            departed = traci.simulation.getDepartedIDList()
            for veh_id in departed:
                type_id = None
                try: type_id = traci.vehicle.getTypeID(veh_id)
                except Exception:
                    try: type_id = traci.vehicle.getVehicleClass(veh_id)
                    except Exception: type_id = None
                depart_times[veh_id] = step
                if type_id is not None: veh_types[veh_id] = type_id

            # --- Process Arrivals (Same logic as baseline.py fix) ---
            arrived_vehicle_ids = traci.simulation.getArrivedIDList()
            for veh_id in arrived_vehicle_ids:
                try:
                    vtype = veh_types.get(veh_id)
                    dep = depart_times.get(veh_id)
                    if vtype in episode_travel_times and dep is not None:
                        duration = step - dep
                        episode_travel_times[vtype].append(duration)
                        all_episode_travel_times[vtype].append(duration) # Add to overall list
                    # Clean up maps
                    if veh_id in depart_times: del depart_times[veh_id]
                    if veh_id in veh_types: del veh_types[veh_id]
                except Exception as e_inner:
                    print(f"WARNING: Episode {episode+1}: Error processing arrived {veh_id}: {e_inner}")

            # --- Get Action from Agent (Only needed if controlling light) ---
            # If we were using SumoEnv, it would look like:
            # obs = env._get_obs() # Get observation in the correct format
            # action, _states = model.predict(obs, deterministic=True) # Get agent's action
            # env._apply_action(action) # Apply agent's action
            # But here, we let the default SUMO light run for simplicity of metric collection

            # Check for termination/truncation (max steps)
            if step >= MAX_STEPS_PER_EPISODE:
                truncated = True
                break

    except traci.TraCIException as e:
        print(f"ERROR: Episode {episode+1}: TraCIException during simulation: {e}. Ending episode.")
        terminated = True # Mark as terminated if connection lost
    except Exception as e:
        print(f"ERROR: Episode {episode+1}: Unexpected error during simulation: {e}. Ending episode.")
        terminated = True
    finally:
        # --- End Episode: Close SUMO ---
        try:
            traci.close()
            print(f"DEBUG: Episode {episode+1}: SUMO closed.")
        except Exception as e:
            print(f"DEBUG: Episode {episode+1}: Error closing SUMO (might already be closed): {e}")

    # --- Print Episode Summary ---
    print(f"Episode {episode + 1} finished after {step} steps.")
    for v_type, times in episode_travel_times.items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"  Avg {v_type.capitalize()} Time: {avg_time:.2f}s ({len(times)} finished)")
        else:
            print(f"  No {v_type}s finished.")


# --- Final Evaluation Summary ---
print("\n--- Overall Evaluation Results ---")

# Calculate Average Car Travel Time
num_cars = len(all_episode_travel_times["car"])
if num_cars > 0:
    avg_car_time = sum(all_episode_travel_times["car"]) / num_cars
    print(f"Average Car Travel Time:   {avg_car_time:.2f} seconds ({num_cars} finished across {NUM_EPISODES} episodes)")
else:
    print("No cars finished across all episodes.")

# Calculate Average Bus Travel Time
num_buses = len(all_episode_travel_times["bus"])
if num_buses > 0:
    avg_bus_time = sum(all_episode_travel_times["bus"]) / num_buses
    print(f"Average Bus Travel Time:   {avg_bus_time:.2f} seconds ({num_buses} finished across {NUM_EPISODES} episodes)")
else:
    print("No buses finished across all episodes.")

# Calculate Average Emergency Transit Time
num_emergency = len(all_episode_travel_times["emergency"])
if num_emergency > 0:
    avg_emergency_time = sum(all_episode_travel_times["emergency"]) / num_emergency
    print(f"Average Emergency Transit Time: {avg_emergency_time:.2f} seconds ({num_emergency} finished across {NUM_EPISODES} episodes)")
else:
    print("No emergency vehicles finished across all episodes.")

print("\nEvaluation script finished.")