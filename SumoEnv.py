import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import time

# --- Constants (MODIFIED for Route File Sync) ---
TRAFFIC_LIGHT_ID = "cluster_10560858054_11707402955_11707402956_11707460716_#5more"
PHASE_NS_GREEN = "GGrrrrGGrrrr"
PHASE_NS_YELLOW = "yyrrrryyrrrr"
PHASE_EW_GREEN = "rrGGrrrrGGrr"
PHASE_EW_YELLOW = "rryyrrrryyrr"

# --- CRITICAL FIXES FOR SYNCHRONIZATION AND EFFICIENCY ---
ROUTE_BEGIN_TIME = 28800.0 # Match the 'depart' time of your first vehicle
ACTION_DURATION = 10        # Agent's action lasts for 10 simulation seconds
SIM_STEP_LENGTH = 1.0       # SUMO runs on 1.0 second steps
# MAX_EPISODE_STEPS = 120 steps * 10s/step = 20 minutes of agent control
MAX_EPISODE_STEPS_TRAINING = 120 
YELLOW_DURATION = 3

# Global cache for detected lanes (persist across environment instances)
INCOMING_LANES = None
# Global variable for cumulative reward tracking (used in Delta calculation)
CUMULATIVE_WAIT_TIME = {} 

# --- Vehicle Type Normalization ---
def normalize_vehicle_type(vtype: str) -> str:
    """
    Normalize vehicle types to standard categories.
    Maps all variations to: car, bus, emergency, auto, motorcycle, truck
    """
    if vtype in ["ambulance", "fire_truck"]:
        return "emergency"
    elif vtype in ["default_car"]:
        return "car"
    elif vtype in ["motorcycle"]:
        return "motorcycle"
    elif vtype in ["truck"]:
        return "truck"
    elif vtype in ["bus"]:
        return "bus"
    elif vtype in ["auto", "taxi"]:
        return "auto"
    elif vtype in ["car", "passenger"]:
        return "car"
    else:
        return vtype  # Return as-is if not recognized


class SumoEnv(gym.Env):
    """
    Custom Gymnasium Environment for SUMO Traffic Signal Control.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, use_gui=False, sumocfg_file="map.sumocfg"):
        super().__init__()
        self.use_gui = use_gui
        self.sumocfg_file = sumocfg_file
        self.episode = 0
        self.current_step = 0
        
        # Use fixed constants
        self.max_episode_steps = MAX_EPISODE_STEPS_TRAINING
        self.sim_time_step = SIM_STEP_LENGTH
        self.action_duration = ACTION_DURATION
        
        # --- Action Space (No Change) ---
        self.action_space = spaces.Discrete(2)
        
        # --- Initialize observation space (Retaining previous logic) ---
        global INCOMING_LANES
        self.num_lanes = len(INCOMING_LANES) if INCOMING_LANES is not None else 4
        self._update_observation_space()
        
        # --- SUMO Setup (No Change) ---
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            if tools not in sys.path:
                sys.path.append(tools)
        else:
            sys.exit("Please declare the 'SUMO_HOME' environment variable.")
        
        if self.use_gui:
            self.sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
            self.render_mode = "human"
        else:
            self.sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
            self.render_mode = None
        
        self.traci_conn = None
        self.detection_distance = 100
        self._departed_buffer = []
        self._arrived_buffer = []
        self.last_total_wait_time = 0.0 # State for Delta Reward
        
        # --- Weighted Reward Tracking (for vehicle priority) ---
        self.last_weighted_wait = 0.0  # Track weighted wait time across all vehicles
        self.last_emergency_wait = 0.0  # Track emergency vehicle wait time
        self.last_truck_wait = 0.0     # Track truck vehicle wait time
        self.last_car_wait = 0.0       # Track car vehicle wait time

    # --- _update_observation_space and _detect_incoming_lanes methods (Enhanced with vehicle counts) ---
    def _update_observation_space(self):
        # Observation space:
        # - Queue lengths per lane (num_lanes)
        # - Phase indicator (1)
        # - Emergency vehicle counts per lane (num_lanes)
        # - Bus vehicle counts per lane (num_lanes)
        # Total: 3*num_lanes + 1
        obs_size = 3 * self.num_lanes + 1
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * obs_size, dtype=np.float32),
            high=np.array([np.inf] * obs_size, dtype=np.float32),
            shape=(obs_size,),
            dtype=np.float32,
        )
        print(f"DEBUG: Observation space updated. Num lanes: {self.num_lanes}, Obs shape: {self.observation_space.shape}")

    def _detect_incoming_lanes(self):
        global INCOMING_LANES
        # (Contains detection logic from previous message - omitted for brevity)
        if INCOMING_LANES is not None:
             self.num_lanes = len(INCOMING_LANES)
             self._update_observation_space()
             return
        
        try:
             known_tls = list(self.traci_conn.trafficlight.getIDList())
             tl_id = TRAFFIC_LIGHT_ID if TRAFFIC_LIGHT_ID in known_tls else known_tls[0] if known_tls else None
             if tl_id:
                 incoming = list(self.traci_conn.trafficlight.getControlledLanes(tl_id))
                 if incoming:
                     INCOMING_LANES = incoming
                     self.num_lanes = len(INCOMING_LANES)
                     self._update_observation_space()
        except Exception:
             pass

    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new episode.
        """
        super().reset(seed=seed)
        self.episode += 1
        self.current_step = 0
        self.last_total_wait_time = 0.0
        # Reset vehicle-specific tracking
        self.last_weighted_wait = 0.0
        self.last_emergency_wait = 0.0
        self.last_truck_wait = 0.0
        self.last_car_wait = 0.0
        global CUMULATIVE_WAIT_TIME
        CUMULATIVE_WAIT_TIME.clear()
        print(f"DEBUG: Resetting environment for Episode {self.episode}...")
        
        # --- Clean up TraCI ---
        if self.traci_conn is not None:
            try:
                self.traci_conn.close()
            except Exception:
                pass
            self.traci_conn = None
        try:
            traci.close()
            print("DEBUG: Ensured TraCI shutdown.")
        except Exception:
            pass
        time.sleep(0.5) # Wait half a second for port release

        # --- Start SUMO simulation (CRITICAL START TIME MODIFICATION) ---
        random_seed = self.episode if self.episode > 0 else 1
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumocfg_file,
            "--no-step-log=true",
            "--no-warnings=true",
            "--quit-on-end=true",
            f"--seed={random_seed}",
            f"--step-length={self.sim_time_step}", # Enforce 1 second simulation steps
            f"--begin={ROUTE_BEGIN_TIME}" # START SIMULATION AT 28800.0
        ]
        
        try:
            traci.start(sumo_cmd)
            self.traci_conn = traci
            print("DEBUG: SUMO started successfully via TraCI.")
            self._detect_incoming_lanes()
        except Exception as e:
            print(f"ERROR: Failed to start SUMO: {e}")
            raise RuntimeError("Could not start SUMO.")
        
        # Initialize buffers (No change)
        self._departed_buffer = []
        self._arrived_buffer = []
        
        observation = self._get_obs()
        info = {}
        print(f"DEBUG: Reset complete. Observation shape: {observation.shape}")
        return observation, info

    def step(self, action):
        """
        Applies an action and steps the simulation for ACTION_DURATION seconds.
        """
        if self.traci_conn is None:
            raise RuntimeError("Traci connection is not alive. Did you call reset()?")
        
        self.current_step += 1
        
        # --- 1. Apply Action ---
        self._apply_action(action)
        
        # --- 2. Step Simulation for fixed duration ---
        steps_to_run = int(self.action_duration / self.sim_time_step) 
        
        current_phase = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
        simulation_running = True
        
        for i in range(steps_to_run):
            try:
                # CRITICAL: Check if loaded vehicles are finished (prevents hang)
                if self.traci_conn.simulation.getMinExpectedNumber() <= 0:
                    simulation_running = False
                    break
                
                self.traci_conn.simulationStep()
                
                # Collect departures and arrivals (No Change)
                try:
                    self._departed_buffer.extend(self.traci_conn.simulation.getDepartedIDList())
                    self._arrived_buffer.extend(self.traci_conn.simulation.getArrivedIDList())
                except Exception:
                    pass
                
                # Yellow phase check: If phase changes from green/yellow to next green/yellow,
                # we should try to complete the required YELLOW_DURATION.
                new_phase = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                if new_phase != current_phase and new_phase in [0, 2]:
                    # Break the loop if the main green phase starts, but only if it's not the last step
                    if i < steps_to_run - 1:
                        pass
                
                if new_phase in [1, 3] and self.traci_conn.trafficlight.getPhaseDuration(TRAFFIC_LIGHT_ID) <= self.sim_time_step:
                    # If it's a yellow phase and it's about to end, let it end naturally.
                    pass
                    
            except traci.TraCIException:
                simulation_running = False
                break
            except Exception:
                simulation_running = False
                break
        
        # --- 3. Get Observation, Reward, Done, Info ---
        terminated = False
        truncated = False
        reward = 0.0
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        if simulation_running:
            try:
                observation = self._get_obs()
                reward = self._get_reward()
                
                min_expected = self.traci_conn.simulation.getMinExpectedNumber()
                # Terminate if all expected vehicles are done AND loaded vehicles were present
                loaded_vehicles = self.traci_conn.simulation.getLoadedNumber()
                terminated = (min_expected <= 0) and (loaded_vehicles > 0)
                truncated = self.current_step >= self.max_episode_steps
            except traci.TraCIException:
                terminated = True
            except Exception:
                terminated = True
        else:
            terminated = True
        
        info = {}
        return observation, reward, terminated, truncated, info

    def _apply_action(self, action):
        """
        Interprets the agent's action (0: Keep, 1: Change) and controls the traffic light.
        """
        try:
            current_phase_index = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            is_green = current_phase_index in [0, 2]
            
            if action == 1 and is_green:
                next_yellow_phase = (current_phase_index + 1) % 4
                self.traci_conn.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_yellow_phase)
                # Set the duration of the yellow light
                self.traci_conn.trafficlight.setPhaseDuration(TRAFFIC_LIGHT_ID, YELLOW_DURATION) 
        except traci.TraCIException:
            pass
        except Exception:
            pass

    def _get_obs(self):
        # Enhanced observation with emergency and bus counts per lane
        global INCOMING_LANES
        if INCOMING_LANES is None or len(INCOMING_LANES) == 0:
             return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        queue_lengths = [0.0] * len(INCOMING_LANES)
        emergency_counts = [0.0] * len(INCOMING_LANES)
        bus_counts = [0.0] * len(INCOMING_LANES)
        phase_indicator = -1.0
        
        try:
             junction_pos = self.traci_conn.junction.getPosition(TRAFFIC_LIGHT_ID)  # type: ignore
             current_phase_index = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)  # type: ignore
             if current_phase_index == 0:
                 phase_indicator = 0.0
             elif current_phase_index == 2:
                 phase_indicator = 1.0
             
             for i, lane_id in enumerate(INCOMING_LANES):
                 queue_lengths[i] = float(self.traci_conn.lane.getLastStepHaltingNumber(lane_id))  # type: ignore
                 vehicles_on_lane = self.traci_conn.lane.getLastStepVehicleIDs(lane_id)  # type: ignore
                 
                 for veh_id in vehicles_on_lane:
                     try:
                         veh_type_id = str(self.traci_conn.vehicle.getTypeID(veh_id))  # type: ignore
                         if veh_type_id in ["ambulance", "fire_truck", "emergency"]:
                             emergency_counts[i] += 1.0
                         elif veh_type_id == "bus":
                             bus_counts[i] += 1.0
                     except:
                         continue
        except Exception:
             pass

        observation = np.concatenate([
            queue_lengths,
            [phase_indicator],
            emergency_counts,
            bus_counts
        ]).astype(np.float32)
        
        return observation

    def _get_reward(self):
        """
        Multi-objective reward with vehicle type prioritization.
        
        Priority weights (highest to lowest):
        - Emergency (ambulance/fire_truck): 5.0x (highest priority)
        - Truck: 4.0x
        - Car (default_car): 3.0x
        - Auto/Taxi: 2.0x
        - Motorcycle: 1.0x
        - Bus: 0.5x (lowest priority)
        """
        # Vehicle type weight mapping - based on user priorities
        vehicle_weights = {
            "emergency": 5.0,
            "ambulance": 5.0,
            "fire_truck": 5.0,
            "truck": 4.0,
            "car": 3.0,
            "default_car": 3.0,
            "passenger": 3.0,
            "auto": 2.0,
            "taxi": 2.0,
            "motorcycle": 1.0,
            "bus": 0.5,
        }
        
        global INCOMING_LANES
        if INCOMING_LANES is None or len(INCOMING_LANES) == 0:
            return 0.0
        
        if self.traci_conn is None:
            return 0.0
        
        current_weighted_wait = 0.0
        current_emergency_wait = 0.0
        current_truck_wait = 0.0
        current_car_wait = 0.0
        
        try:
            all_veh_ids = self.traci_conn.vehicle.getIDList()  # type: ignore
            
            for veh_id in all_veh_ids:
                try:
                    wait_time_sec = float(self.traci_conn.vehicle.getAccumulatedWaitingTime(veh_id))  # type: ignore
                    vtype = str(self.traci_conn.vehicle.getTypeID(veh_id))  # type: ignore
                    # Normalize vehicle types
                    vtype = normalize_vehicle_type(vtype)
                    
                    # Get weight for this vehicle type
                    weight = vehicle_weights.get(vtype, 1.0)
                    weighted_wait = wait_time_sec * weight
                    
                    # Track by category for multi-objective optimization
                    current_weighted_wait += weighted_wait
                    if vtype == "emergency":
                        current_emergency_wait += wait_time_sec
                    elif vtype == "truck":
                        current_truck_wait += wait_time_sec
                    elif vtype == "car":
                        current_car_wait += wait_time_sec
                    
                except (traci.TraCIException, ValueError, TypeError):
                    continue
            
            # Calculate deltas (improvements)
            delta_weighted = current_weighted_wait - self.last_weighted_wait
            delta_emergency = current_emergency_wait - self.last_emergency_wait
            delta_truck = current_truck_wait - self.last_truck_wait
            delta_car = current_car_wait - self.last_car_wait
            
            # Multi-objective reward based on priority: emergency > truck > car > others
            # More balanced: 45% general, 30% emergency, 15% truck, 10% car
            reward = (
                -0.45 * (delta_weighted / 1000.0) +      # General flow optimization
                -0.30 * (delta_emergency / 500.0) +      # Emergency priority (highest)
                -0.15 * (delta_truck / 1000.0) +         # Truck priority
                -0.10 * (delta_car / 1000.0)             # Car priority
            )
            
            # Update state
            self.last_weighted_wait = current_weighted_wait
            self.last_emergency_wait = current_emergency_wait
            self.last_truck_wait = current_truck_wait
            self.last_car_wait = current_car_wait
            self.last_bus_wait = current_bus_wait
            
            reward = float(reward)
        except traci.TraCIException:
            return 0.0
        except Exception:
            return 0.0
        
        return reward
    
    def pop_departed(self):
        deps = list(self._departed_buffer)
        self._departed_buffer.clear()
        return deps

    def pop_arrived(self):
        arrs = list(self._arrived_buffer)
        self._arrived_buffer.clear()
        return arrs
    
    def render(self):
        pass

    def close(self):
        if self.traci_conn is not None:
            try:
                self.traci_conn.close()
            except Exception:
                pass
            finally:
                self.traci_conn = None
        try:
            traci.close()
        except Exception:
            pass