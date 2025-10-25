# Adaptive Traffic Signal Control using Multi-Objective Actor-Critic RL

## 🔹 Project Goal

This project aims to develop an intelligent traffic signal controller using Reinforcement Learning (RL) to adaptively manage a simulated intersection. The primary goal is to **reduce overall traffic congestion** while strategically **prioritizing emergency vehicles (ambulances) and public transport (buses)**, inspired by traffic challenges in urban environments like Bengaluru.
---

## 🛠️ Approach

1.  [cite_start]**Simulation Environment:** We use **SUMO (Simulation of Urban Mobility)** [cite: 7] to create a traffic environment.
    * **Phase 1 (Core):** A synthetic 4-way intersection generated using `netgenerate`. Traffic includes cars, buses, and emergency vehicles spawned using `<flow>` definitions.
    * **Phase 2 (Extension):** Potential porting to a real-world Bengaluru intersection map (e.g., Silk Board) using OpenStreetMap data.
2.  [cite_start]**Reinforcement Learning Agent:** A **Proximal Policy Optimization (PPO)** agent [cite: 14] from the **Stable Baselines3** library learns the control policy.
3.  **Multi-Objective Reward Function:** The agent is trained to optimize a weighted reward function that balances:
    * Minimizing average vehicle travel/wait time (efficiency).
    * Maximizing throughput.
    * **Heavily rewarding** the quick passage of emergency vehicles.
    * **Rewarding** reduced waiting times for buses.
4.  [cite_start]**(Potential Extensions):** Based on the original plan, future work could include Curriculum Learning and Explainable RL (XRL)[cite: 15, 16].

---

## ⚙️ Setup & Installation

1.  **Prerequisites:**
    * **SUMO:** Install SUMO (version 1.19.0 or compatible). Make sure the `SUMO_HOME` environment variable is set and SUMO binaries are in your system PATH.
    * **Python:** Python 3.8+ is required.
2.  **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd sumo-traffic-rl-project
    ```
3.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    # Activate (Linux/macOS)
    source venv/bin/activate
    # Activate (Windows PowerShell)
    .\venv\Scripts\Activate.ps1
    # Activate (Windows CMD)
    .\venv\Scripts\activate.bat
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Running the Project

*(Ensure your virtual environment is activated)*

1.  **Run the Baseline (Fixed-Time Controller):**
    This script runs the simulation with the default SUMO traffic light program and reports performance metrics.
    ```bash
    python baseline.py
    ```
    *(Note: Edit `SUMO_BINARY` in `baseline.py` to `sumo-gui` if you want to watch the simulation, or `sumo` for faster headless runs.)*

2.  **Train the PPO Agent:**
    *(This script will be created in Task 2.4)*
    ```bash
    # Example command (details TBD)
    python train.py --total-timesteps 100000 --save-path models/ppo_traffic
    ```

3.  **Evaluate a Trained Agent:**
    *(This script might be created later)*
    ```bash
    # Example command (details TBD)
    python evaluate.py --model-path models/ppo_traffic.zip --num-episodes 10
    ```

---

## 📁 File Structure (Key Files)
