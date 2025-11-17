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
    def __init__(self, use_gui=False, sumocfg_file="map.sumocfg2", network_type="default"):
        """
        Initializes the SUMO environment.
        
        Args:
            use_gui (bool): If True, use sumo-gui; if False, use sumo (headless)
            sumocfg_file (str): Name of the SUMO config file
            network_type (str): Type of network - "default" (A1) or "mg_road"
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple possible locations for the config file
        # Priority depends on network type
        if network_type == "mg_road":
            # For MG Road, prioritize SUMO_Trinity_Traffic_sim folder
            possible_paths = [
                os.path.join(script_dir, "SUMO_Trinity_Traffic_sim", sumocfg_file),  # Trinity first (priority)
                os.path.join(script_dir, sumocfg_file),  # Root folder
                os.path.join(script_dir, "osm_sudo_map_2", sumocfg_file),  # osm_sudo_map_2 as fallback
            ]
        else:
            # For default network, prioritize root folder with map.sumocfg2
            possible_paths = [
                os.path.join(script_dir, sumocfg_file),  # Root folder
                os.path.join(script_dir, "osm_sudo_map_2", sumocfg_file),  # osm_sudo_map_2 subfolder
                os.path.join(script_dir, "SUMO_Trinity_Traffic_sim", sumocfg_file),  # Trinity as fallback
            ]
        
        config_file = None
        for path in possible_paths:
            if os.path.exists(path):
                config_file = path
                print(f"DEBUG: Found config file at {config_file}")
                break
        
        if config_file is None:
            raise FileNotFoundError(
                f"Config file '{sumocfg_file}' not found in any of these locations:\n" +
                "\n".join(possible_paths)
            )
        
        super().__init__()
        self.use_gui = use_gui
        self.sumocfg_file = config_file  # Store the full absolute path
        self.network_type = network_type
        
        # Network-specific configuration
        if network_type == "mg_road":
            # MG Road network - will detect traffic lights at runtime
            self.traffic_light_ids = []
            self.incoming_lanes = []
        else:
            # Default network with A1 traffic light
            self.traffic_light_ids = ["A1"]
            self.incoming_lanes = ["B2A1_0", "B1A1_0", "B4A1_0", "B3A1_0"]
        
        self.episode = 0 # Track episode count
        self.current_step = 0 # Track steps within an episode
        self.max_episode_steps = 3600 # Define a maximum episode length

        # --- Action Space (Unchanged) ---
        self.action_space = spaces.Discrete(2) #

        # --- Observation Space ---
        # Use a fixed observation size that works for any network
        obs_size = 13  # Generic size that accommodates most networks
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * obs_size, dtype=np.float32),
            high=np.array([np.inf] * obs_size, dtype=np.float32),
            shape=(obs_size,),
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
        # Buffers to collect depart/arrival events during the internal stepping loop
        self._departed_buffer = []
        self._arrived_buffer = []
    # --- END __init__ ---

    def _get_traffic_lights_for_network(self):
        """Detect available traffic lights in the network after SUMO starts"""
        # This will be called during reset when TraCI connection is ready
        return []

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

        # --- Start SUMO simulation with random seed for stochastic flows ---
        # Use episode number as seed to ensure different flows per episode
        random_seed = self.episode if self.episode > 0 else 1
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumocfg_file,
            "--no-step-log=true",
            "--no-warnings=true",
            "--quit-on-end=true",
            f"--seed={random_seed}"  # Enable stochastic vehicle generation
        ]
        try:
            traci.start(sumo_cmd)
            self.traci_conn = traci
            print("DEBUG: SUMO started successfully via TraCI.")
            
            # Detect traffic lights and lanes if using MG Road
            if self.network_type == "mg_road":
                try:
                    available_tls = self.traci_conn.trafficlight.getIDList()
                    if available_tls:
                        self.traffic_light_ids = available_tls
                        print(f"DEBUG: Found traffic lights in network: {available_tls}")
                        
                        # Get incoming lanes for the first traffic light
                        tl_id = available_tls[0]
                        try:
                            self.incoming_lanes = list(self.traci_conn.trafficlight.getControlledLanes(tl_id))[:4]
                            print(f"DEBUG: Found incoming lanes: {self.incoming_lanes}")
                        except Exception as e:
                            print(f"WARNING: Could not get controlled lanes: {e}")
                            self.incoming_lanes = []
                except Exception as e:
                    print(f"WARNING: Could not detect traffic lights: {e}")
                    self.traffic_light_ids = []
        except Exception as e:
            print(f"ERROR: Failed to start SUMO with command {' '.join(sumo_cmd)}: {e}")
            raise RuntimeError("Could not start SUMO.")

        # Initialize buffers and capture vehicles already present at simulation start
        self._departed_buffer = []
        self._arrived_buffer = []
        try:
            existing = self.traci_conn.vehicle.getIDList()
            if existing:
                # Treat existing vehicles as departed at time 0 so downstream
                # metric collectors can pair arrivals to depart times.
                self._departed_buffer.extend(existing)
                print(f"DEBUG: Found {len(existing)} vehicles already in simulation at reset; added to departed buffer.")
        except Exception:
            # If vehicle list not available yet, ignore
            pass

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
        tl_id = self.traffic_light_ids[0] if self.traffic_light_ids else "A1"
        current_phase = self.traci_conn.trafficlight.getPhase(tl_id)
        steps_taken = 0
        simulation_running = True

        while self.traci_conn.simulation.getTime() < target_time:
             try:
                  if self.traci_conn.simulation.getMinExpectedNumber() <= 0:
                       print("DEBUG: No vehicles expected, ending step early.")
                       simulation_running = False
                       break
                  self.traci_conn.simulationStep()
                  # Collect departed/arrived vehicles during the stepping loop so
                  # external evaluators don't miss events when multiple internal
                  # steps are executed per env.step()
                  try:
                      departed_now = self.traci_conn.simulation.getDepartedIDList()
                      if departed_now:
                          self._departed_buffer.extend(departed_now)
                      arrived_now = self.traci_conn.simulation.getArrivedIDList()
                      if arrived_now:
                          self._arrived_buffer.extend(arrived_now)
                  except Exception:
                      # Non-fatal: continue without buffering if calls fail
                      pass
                  steps_taken += 1
                  new_phase = self.traci_conn.trafficlight.getPhase(tl_id)
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

    # --- Utility methods for external metric collectors ---
    def pop_departed(self):
        """Return and clear the list of departed vehicle IDs collected since last pop."""
        deps = list(self._departed_buffer)
        self._departed_buffer.clear()
        return deps

    def pop_arrived(self):
        """Return and clear the list of arrived vehicle IDs collected since last pop."""
        arrs = list(self._arrived_buffer)
        self._arrived_buffer.clear()
        return arrs


    def _apply_action(self, action):
        """
        Interprets the agent's action (0: Keep, 1: Change) and controls the traffic light.
        """
        try:
            tl_id = self.traffic_light_ids[0] if self.traffic_light_ids else "A1"
            current_phase_index = self.traci_conn.trafficlight.getPhase(tl_id)
            is_green = current_phase_index in [0, 2]

            if action == 1 and is_green:
                next_yellow_phase = (current_phase_index + 1) % 4
                self.traci_conn.trafficlight.setPhase(tl_id, next_yellow_phase)
                # DEBUG: Log when action is applied
                if self.current_step <= 5 or self.current_step % 50 == 0:
                    print(f"DEBUG [Step {self.current_step}]: Action CHANGE applied. Phase {current_phase_index} -> {next_yellow_phase}")
            else:
                # DEBUG: Log when action is NOT applied
                if action == 1 and not is_green:
                    if self.current_step <= 5 or self.current_step % 50 == 0:
                        print(f"DEBUG [Step {self.current_step}]: Action CHANGE requested but NOT applied (phase {current_phase_index} is yellow)")
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
        # Use incoming lanes or fallback to default
        lanes_to_use = self.incoming_lanes if self.incoming_lanes else ["B2A1_0", "B1A1_0", "B4A1_0", "B3A1_0"]
        tl_to_use = self.traffic_light_ids[0] if self.traffic_light_ids else "A1"
        
        queue_lengths = [0.0] * len(lanes_to_use)
        emergency_approaching = [0.0] * len(lanes_to_use)
        bus_approaching = [0.0] * len(lanes_to_use)
        phase_indicator = -1.0

        try:
             junction_pos = self.traci_conn.junction.getPosition(tl_to_use)
             current_phase_index = self.traci_conn.trafficlight.getPhase(tl_to_use)
             if current_phase_index == 0: phase_indicator = 0.0
             elif current_phase_index == 2: phase_indicator = 1.0

        except (traci.TraCIException, Exception) as e:
             print(f"ERROR: Could not get junction position or phase for {tl_to_use}: {e}. Returning zero observation.")
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        for i, lane_id in enumerate(lanes_to_use):
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

        # Pad or truncate to match observation space
        if len(observation) < self.observation_space.shape[0]:
            observation = np.pad(observation, (0, self.observation_space.shape[0] - len(observation)), mode='constant')
        elif len(observation) > self.observation_space.shape[0]:
            observation = observation[:self.observation_space.shape[0]]

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

        # Use incoming lanes or fallback
        lanes_to_use = self.incoming_lanes if self.incoming_lanes else ["B2A1_0", "B1A1_0", "B4A1_0", "B3A1_0"]
        
        try:
            total_halting_vehicles = 0
            total_waiting_time = 0.0
            bus_wait_reduction = 0.0
            # emergency_wait_penalty = 0.0 # Placeholder if needed later

            for lane_id in lanes_to_use:
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