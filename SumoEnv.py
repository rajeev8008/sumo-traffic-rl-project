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
    Observation Space: Box - Represents queue lengths (halting vehicles) per incoming lane.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30} # Include 'human' for GUI rendering

    def __init__(self, use_gui=False, sumocfg_file="map.sumocfg"):
        """
        Initializes the SUMO environment.
        """
        super().__init__()
        self.use_gui = use_gui
        self.sumocfg_file = sumocfg_file
        self.episode = 0 # Track episode count
        self.current_step = 0 # Track steps within an episode
        self.max_episode_steps = 3600 # Define a maximum episode length (e.g., 1 hour sim time)

        # --- Action Space ---
        # 0: Keep the current green phase
        # 1: Switch to the next green phase (via yellow)
        self.action_space = spaces.Discrete(2)

        # --- Observation Space ---
        # Queue length (halting cars) for each of the 4 incoming lanes
        num_lanes = len(INCOMING_LANES)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(num_lanes,), dtype=np.float32)

        # --- SUMO Setup ---
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            if tools not in sys.path: # Avoid adding duplicate paths
                 sys.path.append(tools)
        else:
            sys.exit("Please declare the 'SUMO_HOME' environment variable.")

        if self.use_gui:
            self.sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
            self.render_mode = "human" # Set render mode if using GUI
        else:
            self.sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
            self.render_mode = None

        self.traci_conn = None # Will hold the TraCI connection


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
        # Run a few initial steps? (Optional, helps populate the network slightly)
        # for _ in range(5):
        #     self.traci_conn.simulationStep()

        observation = self._get_obs()
        info = {} # You can add extra info here if needed

        print(f"DEBUG: Reset complete. Initial observation: {observation}")
        return observation, info


    def step(self, action):
        """
        Applies an action and steps the simulation.
        Returns: observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # --- 1. Apply Action ---
        self._apply_action(action)

        # --- 2. Step Simulation ---
        # Determine how many SUMO steps to run. Let's run until the light changes
        # or a fixed time passes (e.g., 10 seconds).
        target_time = self.traci_conn.simulation.getTime() + 10 # Target 10 seconds in future
        current_phase = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
        steps_taken = 0

        while self.traci_conn.simulation.getTime() < target_time:
             self.traci_conn.simulationStep()
             steps_taken += 1
             new_phase = self.traci_conn.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
             # If phase changed (e.g., yellow finished), stop stepping early
             if new_phase != current_phase and new_phase in [0, 2]: # Check if it landed on a green phase
                  break
             # Break if simulation ended prematurely (e.g., no cars left)
             if self.traci_conn.simulation.getMinExpectedNumber() <= 0:
                  break

        # print(f"DEBUG: Stepped SUMO for {steps_taken} steps.")

        # --- 3. Get Observation, Reward, Done, Info ---
        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self.traci_conn.simulation.getMinExpectedNumber() <= 0 # Episode ends if no vehicles are expected
        truncated = self.current_step >= self.max_episode_steps # Episode ends if max steps reached
        info = {}

        # print(f"DEBUG: Step {self.current_step}, Action: {action}, Reward: {reward}, Term: {terminated}, Trunc: {truncated}")

        return observation, reward, terminated, truncated, info


    def _apply_action(self, action):
        """
        Interprets the agent's action (0: Keep, 1: Change) and controls the traffic light.
        """
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


    def _get_obs(self):
        """
        Retrieves the current state (observation) of the environment.
        Current implementation: Halting vehicles (queue length) for each incoming lane.
        """
        queue_lengths = []
        for lane_id in INCOMING_LANES:
            try:
                # getLastStepHaltingNumber is generally preferred for queue length
                queue = self.traci_conn.lane.getLastStepHaltingNumber(lane_id)
                queue_lengths.append(queue)
            except traci.TraCIException:
                 # This might happen if the simulation ends unexpectedly
                 print(f"WARNING: TraCIException getting halting number for lane {lane_id}. Appending 0.")
                 queue_lengths.append(0)
            except Exception as e:
                 print(f"ERROR: Unexpected error getting halting number for lane {lane_id}: {e}")
                 queue_lengths.append(0) # Append 0 in case of unexpected errors

        # Ensure the observation is a numpy array of the correct type and shape
        observation = np.array(queue_lengths, dtype=np.float32)
        # print(f"DEBUG: Current Observation (Queues): {observation}")
        return observation


    def _get_reward(self):
        """
        Calculates the reward based on the current state.
        Current implementation: Negative sum of queue lengths (aim to minimize queues).
        """
        current_queues = self._get_obs() # Get current queue state again (or use stored value if needed)
        # Make sure reward is a scalar float
        reward = float(-np.sum(current_queues))
        # print(f"DEBUG: Calculated Reward: {reward}")
        return reward


    def render(self):
        """
        Gymnasium requires a render method. In our case, SUMO handles rendering
        if use_gui is True. This method doesn't need to do anything.
        """
        pass # SUMO GUI handles rendering


    def close(self):
        """
        Closes the TraCI connection when the environment is garbage collected.
        """
        if self.traci_conn is not None:
            try:
                self.traci_conn.close()
                print("DEBUG: TraCI connection closed.")
            except Exception as e:
                print(f"DEBUG: Error closing TraCI on env close (might already be closed): {e}")
            finally:
                self.traci_conn = None