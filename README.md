# Adaptive Traffic Signal Control using RL - Phase 1 Setup

## 🔹 Project Goal

This project aims to develop an intelligent traffic signal controller using Reinforcement Learning (RL). This phase focuses on setting up a simulation environment, establishing a baseline, and building the interface for the RL agent.

---

## 🛠️ Current Setup

1.  **Simulation Environment:** We use **SUMO (Simulation of Urban Mobility)**.
    * A synthetic 4-way intersection (`map.net.xml`) with a central traffic light.
    * Traffic includes cars, buses, and emergency vehicles (`traffic.rou.xml`).
2.  **Baseline Controller:** A script (`baseline.py`) runs the simulation with SUMO's default fixed-time light and records performance metrics.
3.  **RL Environment Wrapper:** A basic Gymnasium wrapper (`SumoEnv.py`) provides the interface between SUMO and an RL agent.

---

## ⚙️ Setup & Installation

1.  **Prerequisites:**
    * **SUMO:** Install SUMO (version 1.19.0 or compatible). Ensure `SUMO_HOME` is set and SUMO is in your PATH.
    * **Python:** Python 3.8+ recommended.
2.  **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd sumo-traffic-rl-project
    ```
3.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    # Activate (Linux/macOS): source venv/bin/activate
    # Activate (Windows): .\venv\Scripts\activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Running Scripts

*(Ensure your virtual environment is activated)*

1.  **Run the Baseline Script:**
    Reports performance of the fixed-time controller.
    ```bash
    python baseline.py
    ```
    *(Uses `sumo` (headless) by default. Edit the script to use `sumo-gui` if needed.)*

2.  **Test the Environment Wrapper:**
    Runs the simulation with the `SumoEnv` wrapper using *random actions*. Useful for debugging the wrapper itself.
    ```bash
    python test_env.py
    ```
    *(Uses `sumo-gui` by default. Edit the script to use `sumo` if needed.)*

---

## 📁 File Structure

* `map.net.xml`: SUMO network file
* `traffic.rou.xml`: SUMO route/flow file
* `map.sumocfg`: SUMO configuration file
* `baseline.py`: Script for baseline controller
* `SumoEnv.py`: Gymnasium environment wrapper
* `test_env.py`: Script to test the SumoEnv wrapper
* `requirements.txt`: Python dependencies
* `README.md`: This file
