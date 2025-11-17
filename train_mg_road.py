import os
import sys
from ppo_agent import PPOAgentConfig, create_ppo_model, create_checkpoint_callback
from SumoEnv import SumoEnv

print("=" * 60)
print("Training PPO Agent for MG Road Network")
print("=" * 60)

# Configuration
config = PPOAgentConfig()

print(f"\nConfiguration:")
print(f"  Total Timesteps: {config.TOTAL_TIMESTEPS}")
print(f"  Learning Rate: {config.LEARNING_RATE}")
print(f"  Checkpoint Frequency: {config.CHECKPOINT_FREQ}")
print(f"  Save Path: {config.save_path}")
print(f"  Log Path: {config.log_path}")

# Create environment for MG Road
print("\nCreating SUMO environment for MG Road...")
try:
    env = SumoEnv(
        use_gui=False,
        sumocfg_file="osm.sumocfg",
        network_type="mg_road"
    )
    print("[OK] Environment created successfully")
except Exception as e:
    print(f"[ERROR] Failed to create environment: {e}")
    sys.exit(1)

# Create PPO agent
print("\nInitializing PPO agent...")
try:
    model = create_ppo_model(env, config)
    print("[OK] PPO agent created successfully")
except Exception as e:
    print(f"[ERROR] Failed to create agent: {e}")
    sys.exit(1)

# Create callbacks
print("Setting up callbacks...")
checkpoint_callback = create_checkpoint_callback(config)

# Train agent
print(f"\n{'=' * 60}")
print(f"Starting training for {config.TOTAL_TIMESTEPS} timesteps...")
print(f"{'=' * 60}\n")

try:
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True
    )
    print("\n[OK] Training completed successfully")
except KeyboardInterrupt:
    print("\n[INFO] Training interrupted by user")
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
finally:
    # Save final model
    final_model_path = os.path.join(config.save_path, "ppo_mg_road_final_model.zip")
    try:
        model.save(final_model_path)
        print(f"[OK] Final model saved to {final_model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
    
    # Close environment
    env.close()

print(f"\n{'=' * 60}")
print("Training finished!")
print(f"To view logs: tensorboard --logdir {config.log_path}")
print(f"{'=' * 60}")
