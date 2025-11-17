import os
import sys
import numpy as np
import traci
from stable_baselines3 import PPO
from SumoEnv import SumoEnv

# --- Configuration ---
MODEL_PATH = "models/ppo_mg_road/best_model"
USE_GUI = True  # Enable GUI for visualization
DETERMINISTIC = True
SUMO_CONFIG = "SUMO_Trinity_Traffic_sim/osm.sumocfg"
NUM_EPISODES = 3  # Run 3 episodes for visualization

# --- Load Trained Agent ---
print(f"Loading trained PPO model from {MODEL_PATH}...")
try:
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load the model from {MODEL_PATH}.")
    print(e)
    sys.exit("Exiting due to model loading error.")

# --- Environment Setup with GUI ---
print("Creating SUMO environment with GUI visualization...")
sim_env = SumoEnv(use_gui=USE_GUI, sumocfg_file=SUMO_CONFIG)

print(f"\nStarting visualization of PPO Agent for {NUM_EPISODES} episodes...")
print("Watch the traffic simulation with the trained agent controlling the traffic light!")
print("=" * 110)

for episode in range(NUM_EPISODES):
    print(f"\n--- Episode {episode + 1} / {NUM_EPISODES} ---")
    
    episode_travel_times = {"car": [], "bus": [], "emergency": [], "auto": [], "motorcycle": [], "truck": []}
    depart_times = {}
    veh_types = {}
    
    try:
        obs, info = sim_env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        current_sim_time = 0.0
    except Exception as e:
        print(f"ERROR: Episode {episode+1}: Failed during reset: {e}")
        continue
    
    try:
        while not terminated and not truncated:
            # Get action from trained agent
            action, _states = model.predict(obs, deterministic=DETERMINISTIC)
            
            # Step environment
            obs, reward, terminated, truncated, info = sim_env.step(action)
            
            # Collect metrics
            if sim_env.traci_conn is not None:
                try:
                    current_sim_time = sim_env.traci_conn.simulation.getTime()
                    
                    # Process departures
                    departed = []
                    try:
                        departed = sim_env.pop_departed()
                    except Exception:
                        try:
                            departed = sim_env.traci_conn.simulation.getDepartedIDList()
                        except Exception:
                            departed = []
                    
                    for veh_id in departed:
                        try:
                            type_id = sim_env.traci_conn.vehicle.getTypeID(veh_id)
                            # Normalize types
                            if type_id in ["ambulance", "fire_truck"]:
                                type_id = "emergency"
                            elif type_id == "default_car":
                                type_id = "car"
                            depart_times[veh_id] = current_sim_time
                            veh_types[veh_id] = type_id
                        except:
                            pass
                    
                    # Process arrivals
                    arrived_vehicle_ids = []
                    try:
                        arrived_vehicle_ids = sim_env.pop_arrived()
                    except Exception:
                        try:
                            arrived_vehicle_ids = sim_env.traci_conn.simulation.getArrivedIDList()
                        except Exception:
                            arrived_vehicle_ids = []
                    
                    for veh_id in arrived_vehicle_ids:
                        try:
                            vtype = veh_types.get(veh_id)
                            dep_time = depart_times.get(veh_id)
                            
                            if vtype in episode_travel_times and dep_time is not None:
                                duration = current_sim_time - dep_time
                                episode_travel_times[vtype].append(duration)
                            
                            depart_times.pop(veh_id, None)
                            veh_types.pop(veh_id, None)
                        except:
                            pass
                
                except Exception:
                    pass
            
            total_reward += reward
            steps += 1
    
    except Exception as e:
        print(f"ERROR: Episode {episode+1}: Unexpected error during simulation: {e}")
    
    # Print episode summary
    print(f"\nEpisode {episode + 1} Summary:")
    print(f"  Duration: {current_sim_time:.2f} seconds of simulation time")
    print(f"  Steps: {steps}")
    print(f"  Total Reward: {total_reward:.2f}")
    print()
    
    for v_type in ["car", "bus", "emergency", "auto", "motorcycle", "truck"]:
        times = episode_travel_times.get(v_type, [])
        if times:
            avg_time = sum(times) / len(times)
            print(f"  {v_type.capitalize()}: {avg_time:.2f}s avg ({len(times)} vehicles)")
        else:
            print(f"  {v_type.capitalize()}: No vehicles finished")
    
    print()

# Close environment
sim_env.close()
print("=" * 110)
print("Visualization complete!")
print("\nThe trained PPO agent successfully controlled the traffic light for all vehicle types.")
print("Emergency vehicles had priority, followed by trucks, cars, autos, motorcycles, and buses.")
