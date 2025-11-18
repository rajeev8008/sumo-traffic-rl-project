# MG Road Trinity Traffic Signal Control - Network Analysis & Design

## Network Analysis

### Traffic Lights Identified
The MG Road (Trinity) network contains 3 main traffic lights controlling 3 junctions:

#### TL 1: cluster_10560858054_11707402955_11707402956_11707460716_#5more
- **Type:** Static (fixed-time)
- **Phases:** 6
  - Phase 0: Duration 40s - State: rrrrGGGggGGGgg
  - Phase 1: Duration 3s  - State: rrrryyyyyyyyyy (Yellow)
  - Phase 2: Duration 4s  - State: rrrrrrrrrrrrrr (All Red)
  - Phase 3: Duration 39s - State: GGGgGrrrrrrrrr
  - Phase 4: Duration 3s  - State: yyyyyrrrrrrrrr (Yellow)
  - Phase 5: Duration 1s  - State: rrrrrrrrrrrrrr (All Red)
- **Incoming Lanes:** 
  - 331497720#1_0
  - 331497720#1_1
  - 1092308701#1_0
  - 1092308701#1_1

#### TL 2: cluster_10784153212_11707402967_11707460669_2999494951
- **Type:** Actuated (dynamic)
- **Phases:** 5
  - Phase 0: Duration 41s - State: GGGgGGggrr (minDur: 5s, maxDur: 50s)
  - Phase 1: Duration 3s  - State: yyyyyyyyrr (Yellow)
  - Phase 2: Duration 3s  - State: rrrrrrrrrr (All Red)
  - Phase 3: Duration 40s - State: GrrgGGGrGG (minDur: 5s, maxDur: 50s)
  - Phase 4: Duration 3s  - State: GrrgGGyryy (Mixed)
- **Incoming Lanes:** (Simplified) 1093123624_0

#### TL 3: joinedS_12543779156_12543779157_12543779194_cluster_12006381693_12006381704_3730964461_3820147036
- **Type:** Actuated (dynamic)
- **Phases:** 12 (complex multi-direction control)
- **Incoming Lanes:** (Simplified) 342001778#1_0

---

## RL Environment Design

### State Space (Observation)
Per traffic light, we observe:
1. **Queue Length** - Number of vehicles waiting at incoming lanes (normalized)
2. **Average Wait Time** - Mean waiting time of queued vehicles (seconds)
3. **Emergency Vehicle Count** - Number of ambulances/fire trucks waiting
4. **Bus Count** - Number of buses waiting
5. **Current Phase** - Current traffic light phase (0-indexed)

**Total Observation Vector:** [15 features from 3 TLs + 1 simulation time] = 16 dimensions
- Shape: (16,) dtype: float32

### Action Space
Multi-Discrete action space allowing independent control of each traffic light:
- **TL 1 Action:** Choose phase from {0, 1, 2, 3, 4, 5} - 6 choices
- **TL 2 Action:** Choose phase from {0, 1, 2, 3, 4} - 5 choices  
- **TL 3 Action:** Choose phase from {0, 1, 2, ..., 11} - 12 choices

**Total Action Space Size:** 6 × 5 × 12 = 360 possible combinations

### Reward Function (Multi-Objective)

The reward combines four objectives:

```
Reward = -α₁ * avg_queue_length 
         - α₂ * avg_wait_time 
         + β₁ * emergency_vehicles_serviced 
         + β₂ * buses_serviced
```

Where:
- **α₁ = 0.01** - Weight for queue penalty (minimize congestion)
- **α₂ = 0.005** - Weight for wait time penalty (minimize delay)
- **β₁ = 0.5** - Weight for emergency bonus (HIGH PRIORITY)
- **β₂ = 0.2** - Weight for bus bonus (MEDIUM PRIORITY)

#### Reward Components:
1. **Congestion Minimization** (-0.01 × avg_queue)
   - Penalizes long queues
   - Encourages throughput

2. **Wait Time Minimization** (-0.005 × avg_wait_time)
   - Penalizes excessive waiting
   - Encourages faster vehicle clearance

3. **Emergency Prioritization** (+0.5 × emergency_count)
   - High bonus for emergency vehicles in green phase
   - Encourages turning green when ambulances arrive

4. **Public Transport Prioritization** (+0.2 × bus_count)
   - Medium bonus for bus prioritization
   - Makes public transport faster/more reliable

---

## Implementation Details

### Environment Class: MGRoadEnv

**Location:** `/sumo_mg_road_env.py`

**Key Methods:**

1. **`__init__`** 
   - Initializes SUMO connection parameters
   - Defines traffic light IDs and incoming lanes
   - Sets up action/observation spaces
   - Configures vehicle type detection for reward

2. **`reset()`**
   - Closes previous SUMO instance if running
   - Starts new SUMO simulation
   - Sets initial traffic light phases
   - Returns initial observation

3. **`step(action)`**
   - Sets each traffic light to specified phase
   - Runs one SUMO simulation step
   - Computes reward
   - Returns (obs, reward, terminated, truncated, info)

4. **`_get_observation()`**
   - Queries SUMO for vehicle data per TL
   - Calculates queue lengths, wait times, vehicle counts
   - Returns normalized numpy array

5. **`_compute_reward()`**
   - Calculates multi-objective reward
   - Prioritizes emergency/bus vehicles
   - Penalizes congestion and wait times

### Gymnasium Compatibility
- ✅ Inherits from `gym.Env`
- ✅ Implements required methods: `reset()`, `step()`, `close()`
- ✅ Uses proper observation/action spaces
- ✅ Returns (obs, reward, terminated, truncated, info) tuple

---

## Next Steps

1. **Test Environment** - Run basic episode to verify connectivity
2. **Curriculum Learning Setup** - Design training difficulty progression
3. **PPO Agent Training** - Implement Stable-Baselines3 PPO
4. **Baseline Comparison** - Compare vs fixed-time control
5. **Multi-Objective Analysis** - Evaluate emergency vs congestion tradeoffs

---

## Vehicle Type Detection

**Emergency Vehicles:** `["emergency", "ambulance", "fire"]`
**Public Transport:** `["bus", "coach"]`
**Regular Vehicles:** `["car", "truck", etc]`

These prefixes are used to identify vehicle types from SUMO's vehicle type IDs for reward computation.

---

## Simulation Parameters

- **Max Steps per Episode:** 3600 (1 hour simulation)
- **Simulation Time Step:** 1 second
- **Lanes Monitored:** 4 incoming lanes for TL1 (simplified for TL2, TL3)
- **GUI Mode:** Configurable (True for visualization, False for fast training)

