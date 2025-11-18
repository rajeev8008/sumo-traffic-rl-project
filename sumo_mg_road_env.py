"""
SumoEnv.py - Gymnasium Environment for Trinity MG Road Traffic Signal Control

Multi-Objective RL Environment with:
- Emergency vehicle prioritization
- Bus/public transport prioritization  
- Congestion reduction

Network: MG Road, Bangalore (Trinity Traffic Sim)
SUMO Version: 1.24.0+
"""

import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import warnings

warnings.filterwarnings('ignore')


class MGRoadEnv(gym.Env):
    """
    Gymnasium environment for MG Road traffic signal control.
    
    Traffic Lights in Network:
    1. cluster_10560858054_11707402955_11707402956_11707460716_#5more
    2. cluster_10784153212_11707402967_11707460669_2999494951
    3. joinedS_12543779156_12543779157_12543779194_cluster_12006381693_12006381704_3730964461_3820147036
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, use_gui=False, sumocfg_file="osm.sumocfg", net_file="osm_network.xml", test_routes=False):
        """
        Initialize MG Road SUMO Environment
        
        Args:
            use_gui (bool): Whether to use SUMO GUI
            sumocfg_file (str): SUMO configuration file name
            net_file (str): Network file name
            test_routes (bool): If True, use test_sumocfg with immediate vehicle generation
        """
        super().__init__()
        
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Use test config if requested
        if test_routes:
            sumocfg_file = "test.sumocfg"
        
        # Try multiple possible locations
        possible_paths = [
            os.path.join(script_dir, "..", "SUMO_Trinity_Traffic_sim", sumocfg_file),
            os.path.join(script_dir, "SUMO_Trinity_Traffic_sim", sumocfg_file),
            os.path.join(script_dir, sumocfg_file),
        ]
        
        self.sumocfg_file = None
        for path in possible_paths:
            if os.path.exists(path):
                self.sumocfg_file = os.path.abspath(path)
                break
        
        if not self.sumocfg_file:
            raise FileNotFoundError(f"Could not find {sumocfg_file} in {script_dir}")
        
        print(f"DEBUG: Found config file at {self.sumocfg_file}")
        
        # SUMO Configuration
        self.use_gui = use_gui
        self.sumo_cmd = ["sumo-gui" if use_gui else "sumo", "-c", self.sumocfg_file]
        self.sumo_cmd.extend(["--no-step-log=true", "--no-warnings=true", "--quit-on-end=true"])
        
        # Traffic Light Configuration
        self.traffic_lights = [
            "cluster_10560858054_11707402955_11707402956_11707460716_#5more",      # TL 1 (6 phases)
            "cluster_10784153212_11707402967_11707460669_2999494951",              # TL 2 (5 phases)
            "joinedS_12543779156_12543779157_12543779194_cluster_12006381693_12006381704_3730964461_3820147036"  # TL 3 (12 phases)
        ]
        
        # Define phases for each traffic light
        # Each traffic light can be in a specific phase (0-indexed)
        self.phases_per_tl = {
            self.traffic_lights[0]: 6,  # TL1: 6 phases
            self.traffic_lights[1]: 5,  # TL2: 5 phases
            self.traffic_lights[2]: 12, # TL3: 12 phases
        }
        
        # Incoming lanes will be discovered dynamically at runtime
        self.incoming_lanes = {}
        
        # Vehicle type detection for multi-objective reward
        self.emergency_vtype_starts = ["emergency", "ambulance", "fire"]
        self.bus_vtype_starts = ["bus", "coach"]
        
        # Action Space: For each TL, choose next phase
        # Total actions = 6 * 5 * 12 = 360 (one action per TL)
        # Using MultiDiscrete to handle each TL independently
        action_dims = [self.phases_per_tl[tl] for tl in self.traffic_lights]
        self.action_space = spaces.MultiDiscrete(action_dims)
        
        # Observation Space
        # Per traffic light: [queue_length, avg_wait_time, emergency_count, bus_count, current_phase]
        # Total: 3 TLs * 5 obs = 15 obs + 1 for simulation time = 16
        num_observations = len(self.traffic_lights) * 5 + 1
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(num_observations,), 
            dtype=np.float32
        )
        
        print(f"[OK] Observation space shape: {self.observation_space.shape}")
        print(f"[OK] Action space: {self.action_space}")
        
        self.sumo_started = False
        self.episode_step = 0
        self.max_steps = 3600  # 1 hour simulation
        
    def _start_sumo(self):
        """Start SUMO simulation"""
        if self.sumo_started:
            return
        
        try:
            # Make sure no other SUMO connection is active
            try:
                traci.close()
            except:
                pass
            
            traci.start(self.sumo_cmd)
            self.sumo_started = True
        except Exception as e:
            raise RuntimeError(f"Failed to start SUMO: {e}")
    
    def _discover_incoming_lanes(self):
        """Discover incoming lanes for each traffic light from network topology"""
        for tl in self.traffic_lights:
            try:
                # Get controlled lanes from traffic light
                controlled_lanes = traci.trafficlight.getControlledLanes(tl)
                
                # Filter for incoming lanes (those that have traffic flowing toward the TL)
                # Usually incoming lanes are those where vehicles wait before the TL
                incoming = []
                for lane in controlled_lanes:
                    try:
                        # Try to access the lane - if it exists in SUMO, add it
                        traci.lane.getLength(lane)
                        incoming.append(lane)
                    except:
                        # Lane doesn't exist, skip
                        continue
                
                if incoming:
                    self.incoming_lanes[tl] = incoming
                    print(f"DEBUG: TL {tl[:30]}... has {len(incoming)} incoming lanes")
                else:
                    # Fallback: use all controlled lanes if filtering fails
                    self.incoming_lanes[tl] = controlled_lanes
                    print(f"DEBUG: TL {tl[:30]}... using {len(controlled_lanes)} controlled lanes")
                    
            except Exception as e:
                print(f"WARNING: Could not discover lanes for {tl[:30]}...: {e}")
                # Fallback: use empty list - we'll handle gracefully in observations
                self.incoming_lanes[tl] = []
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if self.sumo_started:
            traci.close()
            self.sumo_started = False
        
        self.episode_step = 0
        print(f"DEBUG: Resetting environment for Episode...")
        
        self._start_sumo()
        
        # Discover incoming lanes dynamically from network
        self._discover_incoming_lanes()
        
        # Set initial phases
        for tl in self.traffic_lights:
            traci.trafficlight.setPhase(tl, 0)
        
        # Run initial steps to stabilize
        for _ in range(10):
            traci.simulationStep()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """
        Execute one step of environment
        
        Args:
            action: MultiDiscrete action [phase_tl1, phase_tl2, phase_tl3]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        if not self.sumo_started:
            raise RuntimeError("SUMO not started. Call reset() first.")
        
        # Set traffic light phases
        for i, tl in enumerate(self.traffic_lights):
            traci.trafficlight.setPhase(tl, action[i])
        
        # Simulate one step
        traci.simulationStep()
        self.episode_step += 1
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._compute_reward()
        
        # Check termination
        terminated = self.episode_step >= self.max_steps
        truncated = False
        
        info = {
            "step": self.episode_step,
            "total_vehicles": traci.vehicle.getIDCount(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation for each traffic light
        
        Returns:
            np.array: [queue_length_tl1, wait_time_tl1, emergency_tl1, bus_tl1, phase_tl1,
                      queue_length_tl2, wait_time_tl2, emergency_tl2, bus_tl2, phase_tl2,
                      queue_length_tl3, wait_time_tl3, emergency_tl3, bus_tl3, phase_tl3,
                      sim_time]
        """
        obs = []
        
        for tl in self.traffic_lights:
            incoming_lanes = self.incoming_lanes[tl]
            
            # Queue length
            queue_length = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in incoming_lanes)
            
            # Average wait time
            wait_times = []
            for lane in incoming_lanes:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                for veh in vehicle_ids:
                    wait_times.append(traci.vehicle.getWaitingTime(veh))
            avg_wait_time = np.mean(wait_times) if wait_times else 0
            
            # Emergency vehicle count
            emergency_count = 0
            bus_count = 0
            for lane in incoming_lanes:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                for veh in vehicle_ids:
                    try:
                        vtype = traci.vehicle.getTypeID(veh)
                        if any(vtype.startswith(etype) for etype in self.emergency_vtype_starts):
                            emergency_count += 1
                        if any(vtype.startswith(btype) for btype in self.bus_vtype_starts):
                            bus_count += 1
                    except:
                        pass
            
            # Current phase
            current_phase = traci.trafficlight.getPhase(tl)
            
            obs.extend([queue_length, avg_wait_time, emergency_count, bus_count, current_phase])
        
        # Add simulation time
        sim_time = traci.simulation.getTime()
        obs.append(sim_time)
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self) -> float:
        """
        Compute multi-objective reward (AMBULANCE PRIORITY VERSION)
        
        Objectives:
        1. HIGHEST PRIORITY: Minimize ambulance wait time (strong penalty if >0)
        2. Minimize average queue length (soft penalty)
        3. Minimize average wait time (soft penalty)
        4. Maximize ambulance prioritization (HIGHEST bonus)
        5. Maximize emergency vehicle prioritization (high bonus)
        6. Maximize bus prioritization (medium bonus)
        7. Reward smooth traffic flow (throughput bonus)
        8. Penalize excessive congestion
        
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        all_queues = []
        all_wait_times = []
        ambulance_count = 0
        ambulance_wait_times = []
        emergency_count = 0
        bus_count = 0
        total_vehicles = 0
        
        for tl in self.traffic_lights:
            incoming_lanes = self.incoming_lanes[tl]
            
            for lane in incoming_lanes:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                all_queues.append(len(vehicle_ids))
                total_vehicles += len(vehicle_ids)
                
                for veh in vehicle_ids:
                    wait_time = traci.vehicle.getWaitingTime(veh)
                    all_wait_times.append(wait_time)
                    
                    try:
                        vtype = traci.vehicle.getTypeID(veh)
                        
                        # HIGHEST PRIORITY: Ambulances
                        if vtype.startswith("ambulance"):
                            ambulance_count += 1
                            ambulance_wait_times.append(wait_time)
                        # Other emergency vehicles (fire, police)
                        elif any(vtype.startswith(etype) for etype in ["emergency", "fire"]):
                            emergency_count += 1
                        # Public transport
                        elif any(vtype.startswith(btype) for btype in self.bus_vtype_starts):
                            bus_count += 1
                    except:
                        pass
        
        # Reward components (AMBULANCE-FOCUSED)
        avg_queue = np.mean(all_queues) if all_queues else 0
        avg_wait = np.mean(all_wait_times) if all_wait_times else 0
        avg_ambulance_wait = np.mean(ambulance_wait_times) if ambulance_wait_times else 0
        
        # CRITICAL: EXTREMELY HEAVY penalty for ambulance waiting (HIGHEST PRIORITY!)
        # Ambulance waiting is absolutely unacceptable
        ambulance_wait_penalty = -avg_ambulance_wait * 1.0  # 10x stronger! (was 0.5)
        
        # Flow bonus - reward moving vehicles (primary objective!)
        flow_bonus = (total_vehicles / 100.0) * 0.5  # Increased from 0.05 (10x!)
        
        # Queue penalty - penalize accumulation
        queue_penalty = -avg_queue * 0.01  # Increased from 0.001 (10x!)
        
        # Wait time penalty - heavy penalty for overall wait
        wait_penalty = -avg_wait * 0.005  # Increased from 0.0005 (10x!)
        
        # NO BONUS for ambulances - only penalty for waiting!
        # The wait_penalty already covers ambulances, but with MUCH higher coefficient
        ambulance_bonus = 0.0  # REMOVED - was causing gaming behavior
        
        # Emergency vehicle bonus (minimal)
        emergency_bonus = emergency_count * 0.2  # Reduced from 1.0
        
        # Bus bonus (minimal)
        bus_bonus = bus_count * 0.05  # Reduced from 0.1
        
        # Congestion penalty
        congestion_penalty = 0.0
        if avg_queue > 30:
            congestion_penalty = -(avg_queue - 30) * 0.05  # Increased from 0.01 (5x!)
        
        # Combine all components
        # Priority: ambulance_wait_penalty (highest) > ambulance_bonus > emergency > bus > others
        reward = (ambulance_wait_penalty + ambulance_bonus + emergency_bonus + bus_bonus + 
                 queue_penalty + wait_penalty + flow_bonus + congestion_penalty)
        
        return reward
    
    def close(self):
        """Close SUMO connection"""
        if self.sumo_started:
            traci.close()
            self.sumo_started = False


if __name__ == "__main__":
    # Test the environment
    env = MGRoadEnv(use_gui=False)
    print("[OK] Environment created successfully")
    
    obs, info = env.reset()
    print(f"[OK] Environment reset. Initial obs shape: {obs.shape}")
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {info['step']}: Reward={reward:.3f}, Queue={obs[0]:.1f}")
        
        if terminated:
            break
    
    env.close()
    print("[OK] Test completed successfully")
