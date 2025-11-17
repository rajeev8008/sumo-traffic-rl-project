from SumoEnv import SumoEnv # Import your custom environment
import time

print("Creating SumoEnv instance...")
env = SumoEnv(use_gui=True) # Use GUI=True to visually inspect the simulation
print("Resetting environment...")
obs, info = env.reset()
print(f"Initial observation: {obs}")

try:
    # Run the simulation loop for a fixed number of steps
    for step_num in range(200): # Run for 200 agent steps
        # Sample a random action from the environment's action space
        action = env.action_space.sample() # 0 or 1
        print(f"\n--- Step {step_num + 1} ---")
        print(f"Taking action: {action}")

        # Apply the action and get the results
        obs, reward, terminated, truncated, info = env.step(action)

        # Print the results of the step
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")

        # Check if the episode ended (either naturally or due to max steps)
        if terminated or truncated:
            print("Episode finished. Resetting environment...")
            obs, info = env.reset() # Reset for the next episode
            print(f"Initial observation for new episode: {obs}")

        # Pause briefly to make the GUI watchable
        time.sleep(0.1)

finally:
    # Ensure the environment is closed properly, even if errors occur
    print("\nClosing environment...")
    env.close()
    print("Test finished.")