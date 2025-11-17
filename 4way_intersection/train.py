import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try multiple locations for config file - prioritize map.sumocfg2 which has A1 traffic light
possible_configs = [
    os.path.join(SCRIPT_DIR, "map.sumocfg2"),  # Priority: has A1 traffic light
    os.path.join(SCRIPT_DIR, "osm.sumocfg"),
    os.path.join(SCRIPT_DIR, "osm_sudo_map_2", "osm.sumocfg"),
    os.path.join(SCRIPT_DIR, "SUMO_Trinity_Traffic_sim", "osm.sumocfg"),
]

CONFIG_FILE = None
for path in possible_configs:
    if os.path.exists(path):
        CONFIG_FILE = path
        break

if CONFIG_FILE is None:
    print("ERROR: Could not find any config file in known locations")
    exit(1)
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your custom environment
from SumoEnv import SumoEnv # Assuming SumoEnv.py is in the same directory

# --- Configuration ---
SAVE_PATH = "models/ppo_sumo_model" # Directory to save the trained model and checkpoints
LOG_PATH = "logs/"                # Directory for TensorBoard logs
TOTAL_TIMESTEPS = 100000          # Total number of agent steps for training (adjust as needed)
CHECKPOINT_FREQ = 10000           # Save a checkpoint every N steps

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# --- Environment Setup ---
print("Creating SUMO environment for training...")
# Instantiate the environment (use_gui=False for faster training)
env = SumoEnv(use_gui=False, sumocfg_file="map.sumocfg2") #

# It's recommended to check your custom environment follows the Gymnasium API
# check_env(env) # Optional: Can take time, comment out after first check

# --- Agent Training ---
print(f"Starting PPO training for {TOTAL_TIMESTEPS} timesteps...")

# Setup Checkpoint Callback
# This saves the model periodically during training
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=SAVE_PATH,
    name_prefix="ppo_checkpoint",
    save_replay_buffer=True, # Set to False if not using HER or an off-policy algorithm
    save_vecnormalize=True,  # Set to True if using VecNormalize wrapper
)

# Instantiate the PPO agent
# "MlpPolicy" is suitable for vector observations (like yours)
# verbose=1 prints training progress
# tensorboard_log=LOG_PATH enables TensorBoard logging
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_PATH,
    # You might need to tune these hyperparameters later:
    # learning_rate=0.0003,
    # n_steps=2048, # Number of steps to collect per environment before updating policy
    # batch_size=64,
    # n_epochs=10,
    # gamma=0.99, # Discount factor
    # gae_lambda=0.95,
    # clip_range=0.2,
    device="cpu" # Use "cuda" if you have a compatible GPU and PyTorch with CUDA
)

# Train the agent
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        log_interval=1, # Log stats every N updates
        tb_log_name="PPO_SUMO_Run1" # Name for the TensorBoard log
    )
except KeyboardInterrupt:
    print("Training interrupted by user.")
finally:
    # Save the final model
    final_model_path = os.path.join(SAVE_PATH, "ppo_sumo_final_model")
    print(f"Saving final model to {final_model_path}.zip")
    model.save(final_model_path)
    # Close the environment
    env.close()

print("\nTraining finished and model saved.")

# --- Optional: TensorBoard ---
print(f"\nTo view training logs, run:\ntensorboard --logdir {LOG_PATH}")