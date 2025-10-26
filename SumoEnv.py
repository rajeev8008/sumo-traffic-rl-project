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

    # --- UPDATED __init__ for Task 2.2 ---
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
        # 0: Keep the current green phase
        # 1: Switch to the next green phase (via yellow)
        self.action_space = spaces.Discrete(2) #

        # --- Observation Space (UPDATED for Task 2.2) ---
        # We need space for:
        # 1. Queue lengths on 4 incoming lanes
        # 2. Current phase indicator (1 value: 0 for NS_Green, 1 for EW_Green, -1 for Yellow)
        # 3. Emergency vehicle approaching flags (4 lanes)
        # 4. Bus approaching flags (4 lanes)
        num_lanes = len(INCOMING_LANES)
        observation_shape = (
            num_lanes + # Queue lengths
            1 +         # Current phase indicator
            num_lanes + # Emergency approaching flags
            num_lanes   # Bus approaching flags
        ,) # The comma makes it a tuple shape (13,)
        # Define bounds (queues can be large, flags/phase are 0/1 or -1)
        # Using a single Box space for simplicity with Stable Baselines3
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * observation_shape[0], dtype=np.float32), # Set low bound to -1 for phase indicator
            high=np.array([np.inf] * observation_shape[0], dtype=np.float32), # Set high bound high for queues
            shape=observation_shape,
            dtype=np.float32
        )
        print(f"DEBUG: Observation space shape: {self.observation_space.shape}") # Debug print shape

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
        # Store detection distance for priority vehicles
        self.detection_distance = 100 # Detect priority vehicles within 100 meters
    # --- END UPDATED __init__ ---

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
            "--no-step-log=true", # Disable step logging for performance
            "--no-warnings=true", # Suppress warnings during training
            "--quit-on-end=true"  # Ensure SUMO closes properly
            # Add other SUMO options if needed
        ]
        try:
            # Important: Use traci.connect() if starting SUMO externally,
            # or traci.start() to let the script manage the SUMO process.
            traci.start(sumo_cmd)
            self.traci_conn = traci # Store the connection object
            print("DEBUG: SUMO started successfully via TraCI.")
        except Exception as e:
            print(f"ERROR: Failed to start SUMO with command {' '.join(sumo_cmd)}: {e}")
            raise RuntimeError("Could not start SUMO.")

        # --- Initial Observation ---
        observation = self._get_obs()
        info = {} # You can add extra info here if needed

        print(f"DEBUG: Reset complete. Initial observation: {observation}")
        return observation, info


    def step(self, action):
        """
        Applies an action and steps the simulation.
        Returns: observation, reward, terminated, truncated, info
        """
        # Ensure connection is alive
        if self.traci_conn is None:
             raise RuntimeError("Traci connection is not alive. Did you call reset()?")

        self.current_step += 1

        # --- 1. Apply Action ---
        self._apply_action(action)

        # --- 2. Step Simulation ---
        # Determine how many SUMO steps to run. Let's run until the light changes
        # or a fixed time passes (e.g., 10 seconds).
        target_time = self.traci_conn.simulation.getTime() + 10 # Target 10 seconds in future
        current_phase = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
        steps_taken = 0
        simulation_running = True

        while self.traci_conn.simulation.getTime() < target_time:
             # Check if simulation is still running before stepping
             try:
                  # Check number of vehicles; if 0, simulation might end
                  if self.traci_conn.simulation.getMinExpectedNumber() <= 0:
                       print("DEBUG: No vehicles expected, ending step early.")
                       simulation_running = False
                       break
                  self.traci_conn.simulationStep()
                  steps_taken += 1
                  new_phase = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                  # If phase changed (e.g., yellow finished), stop stepping early
                  if new_phase != current_phase and new_phase in [0, 2]: # Check if it landed on a green phase
                       break
             except traci.TraCIException as e:
                  print(f"ERROR: TraCIException during simulationStep: {e}. Assuming simulation ended.")
                  simulation_running = False
                  break
             except Exception as e:
                  print(f"ERROR: Unexpected error during simulationStep: {e}")
                  simulation_running = False
                  break # Exit loop on unexpected error

        # print(f"DEBUG: Stepped SUMO for {steps_taken} steps.")

        # --- 3. Get Observation, Reward, Done, Info ---
        terminated = False
        truncated = False
        reward = 0.0 # Default reward if simulation ended abruptly
        observation = np.zeros(self.observation_space.shape, dtype=np.float32) # Default observation

        if simulation_running:
            try:
                observation = self._get_obs()
                reward = self._get_reward()
                terminated = self.traci_conn.simulation.getMinExpectedNumber() <= 0 # Episode ends if no vehicles are expected
                truncated = self.current_step >= self.max_episode_steps # Episode ends if max steps reached
            except traci.TraCIException as e:
                print(f"ERROR: TraCIException after step loop: {e}. Terminating episode.")
                terminated = True # End episode if traci fails here
                # Use last known good observation or zeros? Let's use zeros.
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            except Exception as e:
                print(f"ERROR: Unexpected error after step loop: {e}. Terminating episode.")
                terminated = True
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
             # Simulation ended during the step loop
             terminated = True

        info = {}

        # print(f"DEBUG: Step {self.current_step}, Action: {action}, Reward: {reward}, Term: {terminated}, Trunc: {truncated}")

        return observation, reward, terminated, truncated, info


    def _apply_action(self, action):
        """
        Interprets the agent's action (0: Keep, 1: Change) and controls the traffic light.
        """
        try:
            current_phase_index = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)

            # Only trigger a change if the light is currently green
            is_green = current_phase_index in [0, 2] # Indices 0 and 2 are green phases

            if action == 1 and is_green:
                # Start transition to the next phase (switch to yellow)
                next_yellow_phase = (current_phase_index + 1) % 4 # 0->1, 2->3
                self.traci_conn.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_yellow_phase)
                # print(f"DEBUG: Action 1 - Switching from Green Phase {current_phase_index} to Yellow Phase {next_yellow_phase}")
            else:
                 # If action is 0, or if currently yellow, do nothing (SUMO handles yellow->green transition)
                 # print(f"DEBUG: Action {action} - Keeping Phase {current_phase_index} (or letting yellow finish)")
                 pass
        except traci.TraCIException as e:
             print(f"WARNING: TraCIException in _apply_action: {e}")
        except Exception as e:
             print(f"ERROR: Unexpected error in _apply_action: {e}")

    # --- UPDATED _get_obs for Task 2.2 ---
    def _get_obs(self):
        """
        Retrieves the current state (observation) of the environment.
        Includes: Halting queues, current phase indicator, priority vehicles approaching.
        """
        queue_lengths = [0.0] * len(INCOMING_LANES)
        emergency_approaching = [0.0] * len(INCOMING_LANES) # Initialize flags to 0
        bus_approaching = [0.0] * len(INCOMING_LANES)       # Initialize flags to 0
        phase_indicator = -1.0 # Default to yellow/unknown

        # Get junction position once
        try:
             junction_pos = self.traci_conn.junction.getPosition(TRAFFIC_LIGHT_ID)
             # Also get phase here to ensure consistency if errors occur later
             current_phase_index = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
             if current_phase_index == 0: # NS Green
                  phase_indicator = 0.0
             elif current_phase_index == 2: # EW Green
                  phase_indicator = 1.0

        except (traci.TraCIException, Exception) as e:
             print(f"ERROR: Could not get junction position or phase for {TRAFFIC_LIGHT_ID}: {e}. Returning zero observation.")
             # Return a zero array of the correct shape if junction info fails
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        for i, lane_id in enumerate(INCOMING_LANES):
            try:
                # 1. Get Queue Length (Halting Vehicles)
                queue_lengths[i] = self.traci_conn.lane.getLastStepHaltingNumber(lane_id)

                # 2. Check for Approaching Priority Vehicles
                vehicles_on_lane = self.traci_conn.lane.getLastStepVehicleIDs(lane_id)

                for veh_id in vehicles_on_lane:
                    try:
                        # Get vehicle position (front bumper x,y)
                        veh_pos = self.traci_conn.vehicle.getPosition(veh_id)
                        # Calculate Euclidean distance
                        distance = np.sqrt((veh_pos[0] - junction_pos[0])**2 + (veh_pos[1] - junction_pos[1])**2)

                        if distance <= self.detection_distance:
                            veh_type_id = self.traci_conn.vehicle.getTypeID(veh_id) # Get vType ID
                            if veh_type_id == "emergency":
                                emergency_approaching[i] = 1.0 # Set flag for this lane
                            elif veh_type_id == "bus":
                                bus_approaching[i] = 1.0 # Set flag for this lane
                    except traci.TraCIException:
                        continue # Skip vehicle if it disappeared (race condition)
                    except Exception as e_dist:
                        print(f"WARNING: Error getting position/distance for {veh_id}: {e_dist}")

            except traci.TraCIException:
                print(f"WARNING: TraCIException getting data for lane {lane_id}. Using 0.")
                # Values already initialized to 0, just continue
                pass
            except Exception as e:
                print(f"ERROR: Unexpected error getting obs for lane {lane_id}: {e}. Using 0.")
                # Values already initialized to 0, just continue
                pass

        # 4. Combine into a single observation vector
        # Order: [QueueN, Q_E, Q_S, Q_W, PhaseIndicator, Em_N, Em_E, Em_S, Em_W, Bus_N, Bus_E, Bus_S, Bus_W]
        observation = np.concatenate([
            queue_lengths,
            [phase_indicator], # Phase indicator needs to be in a list/array
            emergency_approaching,
            bus_approaching
        ]).astype(np.float32)

        # print(f"DEBUG: Current Observation: {observation}") # Optional detailed print
        # Ensure observation matches the defined space shape
        if observation.shape != self.observation_space.shape:
             print(f"ERROR: Observation shape mismatch! Expected {self.observation_space.shape}, got {observation.shape}. Returning zeros.")
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        return observation
    # --- END UPDATED _get_obs ---

    def _get_reward(self):
        """
        Calculates the reward based on the current state.
        Current implementation: Negative sum of queue lengths (aim to minimize queues).
        TODO: Update this in Task 2.3 for multi-objective reward.
        """
        # Get queue lengths from the *current* state (after the step)
        # Slicing the observation array is one way, or call traci again
        # Let's call traci again for simplicity here, though slicing obs might be slightly faster
        reward = 0.0
        try:
             total_halting = 0
             for lane_id in INCOMING_LANES:
                  total_halting += self.traci_conn.lane.getLastStepHaltingNumber(lane_id)
             reward = float(-total_halting)
        except traci.TraCIException as e:
             print(f"WARNING: TraCIException in _get_reward: {e}")
             # Return 0 reward if we can't get state
        except Exception as e:
             print(f"ERROR: Unexpected error in _get_reward: {e}")

        # print(f"DEBUG: Calculated Reward: {reward}")
        return reward


    def render(self):
        """
        Gymnasium requires a render method. SUMO GUI handles rendering.
        """
        pass # Not needed if SUMO handles rendering


    def close(self):
        """
        Closes the TraCI connection.
        """
        if self.traci_conn is not None:
            try:
                self.traci_conn.close()
                print("DEBUG: TraCI connection closed.")
            except Exception as e:
                print(f"DEBUG: Error closing TraCI on env close (might already be closed): {e}")
            finally:
                self.traci_conn = None