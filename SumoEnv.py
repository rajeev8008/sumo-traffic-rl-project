import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci

# --- Constants ---
TRAFFIC_LIGHT_ID = "A1" # From your map.net.xml
# Default phases from netgenerate (match your map.net.xml tlLogic)
PHASE_NS_GREEN = "GGggrrrrGGggrrrr" # North-South Green (index 0)
PHASE_NS_YELLOW = "yyyyrrrryyyyrrrr" # Yellow transition (index 1)
PHASE_EW_GREEN = "rrrrGGggrrrrGGgg" # East-West Green (index 2)
PHASE_EW_YELLOW = "rrrryyyyrrrryyyy" # Yellow transition (index 3)

# Confirmed Incoming Lanes for Junction A1 from your map.net.xml
# Order: North, East, South, West
INCOMING_LANES = ["B2A1_0", "B1A1_0", "B4A1_0", "B3A1_0"] #

YELLOW_DURATION = 3 # Default yellow phase duration in the <tlLogic>
# SIM_STEP_LENGTH = 1.0 # Standard simulation step length (usually 1 second)


class SumoEnv(gym.Env):
    """
    Custom Gymnasium Environment for SUMO Traffic Signal Control.

    Action Space: Discrete(2) - 0: Keep Phase, 1: Change Phase
    Observation Space: Box(13,) - See __init__ and _get_obs for details.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30} # Include 'human' for GUI rendering

    # --- __init__ includes Task 2.2 changes ---
    def __init__(self, use_gui=False, sumocfg_file="map.sumocfg"):
        """
        Initializes the SUMO environment.
        """
        super().__init__()
        self.use_gui = use_gui
        self.sumocfg_file = sumocfg_file
        self.episode = 0 # Track episode count
        self.current_step = 0 # Track steps within an episode
        self.max_episode_steps = 3600 # Define a maximum episode length

        # --- Action Space (Unchanged) ---
        self.action_space = spaces.Discrete(2) #

        # --- Observation Space (From Task 2.2) ---
        num_lanes = len(INCOMING_LANES)
        observation_shape = (
            num_lanes + # Queue lengths
            1 +         # Current phase indicator
            num_lanes + # Emergency approaching flags
            num_lanes   # Bus approaching flags
        ,) # Shape (13,)
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * observation_shape[0], dtype=np.float32),
            high=np.array([np.inf] * observation_shape[0], dtype=np.float32),
            shape=observation_shape,
            dtype=np.float32
        )
        print(f"DEBUG: Observation space shape: {self.observation_space.shape}")

        # --- SUMO Setup ---
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
        self.detection_distance = 100 # Detect priority vehicles within 100 meters
    # --- END __init__ ---

    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new episode.
        """
        super().reset(seed=seed)
        self.episode += 1
        self.current_step = 0
        print(f"DEBUG: Resetting environment for Episode {self.episode}...")

        # --- Close existing TraCI connection if open ---
        if self.traci_conn is not None:
            try:
                self.traci_conn.close()
            except Exception as e:
                print(f"DEBUG: Error closing previous TraCI connection (might already be closed): {e}")
            self.traci_conn = None
            print("DEBUG: Closed existing TraCI connection.")

        # --- Start SUMO simulation ---
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumocfg_file,
            "--no-step-log=true",
            "--no-warnings=true",
            "--quit-on-end=true"
        ]
        try:
            traci.start(sumo_cmd)
            self.traci_conn = traci
            print("DEBUG: SUMO started successfully via TraCI.")
        except Exception as e:
            print(f"ERROR: Failed to start SUMO with command {' '.join(sumo_cmd)}: {e}")
            raise RuntimeError("Could not start SUMO.")

        # --- Initial Observation ---
        observation = self._get_obs()
        info = {}

        print(f"DEBUG: Reset complete. Initial observation: {observation}")
        return observation, info


    def step(self, action):
        """
        Applies an action and steps the simulation.
        Returns: observation, reward, terminated, truncated, info
        """
        if self.traci_conn is None:
             raise RuntimeError("Traci connection is not alive. Did you call reset()?")

        self.current_step += 1

        # --- 1. Apply Action ---
        self._apply_action(action)

        # --- 2. Step Simulation ---
        target_time = self.traci_conn.simulation.getTime() + 10
        current_phase = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
        steps_taken = 0
        simulation_running = True

        while self.traci_conn.simulation.getTime() < target_time:
             try:
                  if self.traci_conn.simulation.getMinExpectedNumber() <= 0:
                       print("DEBUG: No vehicles expected, ending step early.")
                       simulation_running = False
                       break
                  self.traci_conn.simulationStep()
                  steps_taken += 1
                  new_phase = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                  if new_phase != current_phase and new_phase in [0, 2]:
                       break
             except traci.TraCIException as e:
                  print(f"ERROR: TraCIException during simulationStep: {e}. Assuming simulation ended.")
                  simulation_running = False
                  break
             except Exception as e:
                  print(f"ERROR: Unexpected error during simulationStep: {e}")
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
                reward = self._get_reward() # Calls the updated reward function
                terminated = self.traci_conn.simulation.getMinExpectedNumber() <= 0
                truncated = self.current_step >= self.max_episode_steps
            except traci.TraCIException as e:
                print(f"ERROR: TraCIException after step loop: {e}. Terminating episode.")
                terminated = True
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            except Exception as e:
                print(f"ERROR: Unexpected error after step loop: {e}. Terminating episode.")
                terminated = True
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)
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
            else:
                 pass
        except traci.TraCIException as e:
             print(f"WARNING: TraCIException in _apply_action: {e}")
        except Exception as e:
             print(f"ERROR: Unexpected error in _apply_action: {e}")

    # --- _get_obs includes Task 2.2 changes ---
    def _get_obs(self):
        """
        Retrieves the current state (observation) of the environment.
        Includes: Halting queues, current phase indicator, priority vehicles approaching.
        """
        queue_lengths = [0.0] * len(INCOMING_LANES)
        emergency_approaching = [0.0] * len(INCOMING_LANES)
        bus_approaching = [0.0] * len(INCOMING_LANES)
        phase_indicator = -1.0

        try:
             junction_pos = self.traci_conn.junction.getPosition(TRAFFIC_LIGHT_ID)
             current_phase_index = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
             if current_phase_index == 0: phase_indicator = 0.0
             elif current_phase_index == 2: phase_indicator = 1.0

        except (traci.TraCIException, Exception) as e:
             print(f"ERROR: Could not get junction position or phase for {TRAFFIC_LIGHT_ID}: {e}. Returning zero observation.")
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        for i, lane_id in enumerate(INCOMING_LANES):
            try:
                queue_lengths[i] = self.traci_conn.lane.getLastStepHaltingNumber(lane_id)
                vehicles_on_lane = self.traci_conn.lane.getLastStepVehicleIDs(lane_id)

                for veh_id in vehicles_on_lane:
                    try:
                        veh_pos = self.traci_conn.vehicle.getPosition(veh_id)
                        distance = np.sqrt((veh_pos[0] - junction_pos[0])**2 + (veh_pos[1] - junction_pos[1])**2)

                        if distance <= self.detection_distance:
                            veh_type_id = self.traci_conn.vehicle.getTypeID(veh_id)
                            if veh_type_id == "emergency":
                                emergency_approaching[i] = 1.0
                            elif veh_type_id == "bus":
                                bus_approaching[i] = 1.0
                    except traci.TraCIException: continue
                    except Exception as e_dist: print(f"WARNING: Error getting position/distance for {veh_id}: {e_dist}")
            except traci.TraCIException: print(f"WARNING: TraCIException getting data for lane {lane_id}. Using 0."); pass
            except Exception as e: print(f"ERROR: Unexpected error getting obs for lane {lane_id}: {e}. Using 0."); pass

        observation = np.concatenate([
            queue_lengths,
            [phase_indicator],
            emergency_approaching,
            bus_approaching
        ]).astype(np.float32)

        if observation.shape != self.observation_space.shape:
             print(f"ERROR: Observation shape mismatch! Expected {self.observation_space.shape}, got {observation.shape}. Returning zeros.")
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        return observation
    # --- END _get_obs ---

    # --- UPDATED _get_reward for Task 2.3 ---
    def _get_reward(self):
        """
        Calculates the multi-objective reward for the previous action.
        Includes: queue penalty, waiting time penalty, bus priority term.
        Emergency priority is handled implicitly by observation + penalties.
        """
        reward = 0.0
        # --- Tunable Weights ---
        W_QUEUE = 0.5      # Penalty per halting vehicle per second
        W_WAIT = 0.1      # Penalty per second of accumulated wait time
        W_BUS_WAIT = 0.2  # Bonus (reduction in penalty) per second of bus waiting time

        try:
            total_halting_vehicles = 0
            total_waiting_time = 0.0
            bus_wait_reduction = 0.0
            # emergency_wait_penalty = 0.0 # Placeholder if needed later

            for lane_id in INCOMING_LANES:
                total_halting_vehicles += self.traci_conn.lane.getLastStepHaltingNumber(lane_id)
                vehicles_on_lane = self.traci_conn.lane.getLastStepVehicleIDs(lane_id)
                for veh_id in vehicles_on_lane:
                     try:
                          # Accumulate total waiting time across all vehicles
                          wait_time_sec = self.traci_conn.vehicle.getAccumulatedWaitingTime(veh_id)
                          total_waiting_time += wait_time_sec

                          # Check types for specific rewards/penalties
                          if self.traci_conn.vehicle.getSpeed(veh_id) < 0.1: # Only consider waiting vehicles
                               veh_type_id = self.traci_conn.vehicle.getTypeID(veh_id)
                               if veh_type_id == "bus":
                                    bus_wait_reduction += W_BUS_WAIT * wait_time_sec # Accumulate bonus for waiting buses
                               # elif veh_type_id == "emergency":
                               #      emergency_wait_penalty += W_EMERGENCY_WAIT * wait_time_sec # Example: add large penalty if needed
                     except traci.TraCIException:
                          continue # Skip if vehicle vanished

            # Calculate reward components
            # Note: Waiting time and halting count are correlated, consider using only one if needed
            queue_penalty = W_QUEUE * total_halting_vehicles
            wait_penalty = W_WAIT * total_waiting_time

            # Combine: Start with penalties, then add bonus/reduction
            reward = -(queue_penalty + wait_penalty) + bus_wait_reduction # - emergency_wait_penalty

            reward = float(reward) # Ensure scalar float

        except traci.TraCIException as e:
             print(f"WARNING: TraCIException in _get_reward: {e}")
             return 0.0 # Return neutral reward on error
        except Exception as e:
             print(f"ERROR: Unexpected error in _get_reward: {e}")
             return 0.0 # Return neutral reward on error

        # print(f"DEBUG: Calculated Reward: {reward}")
        return reward
    # --- END UPDATED _get_reward ---

    def render(self):
        """ Gymnasium render method """
        pass

    def close(self):
        """ Closes the TraCI connection. """
        if self.traci_conn is not None:
            try:
                self.traci_conn.close()
                print("DEBUG: TraCI connection closed.")
            except Exception as e:
                print(f"DEBUG: Error closing TraCI on env close (might already be closed): {e}")
            finally:
                self.traci_conn = None