# Adaptive Traffic Signal Control using RL - Baseline

## 🔹 Project Goal

This project aims to develop an intelligent traffic signal controller using Reinforcement Learning (RL). This initial phase focuses on setting up a simulation environment and establishing a baseline performance metric using a standard fixed-time controller.

---

## 🛠️ Current Setup

1.  **Simulation Environment:** We use **SUMO (Simulation of Urban Mobility)**.
    * A synthetic 4-way intersection generated using `netgenerate` (`map.net.xml`).
    * Traffic includes cars, buses, and emergency vehicles spawned using `<flow>` definitions (`traffic.rou.xml`).
2.  **Baseline Controller:** A Python script (`baseline.py`) uses `traci` to run the simulation with SUMO's default fixed-time traffic light program and collects performance metrics.

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

## ▶️ Running the Baseline

*(Ensure your virtual environment is activated)*

1.  **Run the Baseline Script:**
    This runs the simulation (without GUI by default) and prints the average travel/transit times.
    ```bash
    python baseline.py
    ```
    *(Note: You can change `SUMO_BINARY = "sumo"` to `SUMO_BINARY = "sumo-gui"` inside `baseline.py` if you want to watch the simulation.)*

---

## 📁 File Structure

* `map.net.xml`: SUMO network file
* `traffic.rou.xml`: SUMO route/flow file
* `map.sumocfg`: SUMO configuration file
* `baseline.py`: Script for baseline controller
* `requirements.txt`: Python dependencies
* `README.md`: This file
