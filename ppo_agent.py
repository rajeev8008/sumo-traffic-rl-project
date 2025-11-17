from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

class PPOAgentConfig:
    """Configuration for PPO agent training"""
    
    # Training parameters
    TOTAL_TIMESTEPS = 100000
    LEARNING_RATE = 3e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    VF_COEF = 0.5
    ENT_COEF = 0.0
    MAX_GRAD_NORM = 0.5
    
    # Checkpoint and logging
    CHECKPOINT_FREQ = 10000
    SAVE_PATH = "models/ppo_mg_road"
    LOG_PATH = "logs/ppo_mg_road"
    
    def __init__(self, project_root=None):
        if project_root is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
        
        self.project_root = project_root
        self.save_path = os.path.join(project_root, self.SAVE_PATH)
        self.log_path = os.path.join(project_root, self.LOG_PATH)
        
        # Create directories
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)


def create_ppo_model(env, config):
    """Create and return a PPO model with specified configuration"""
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        vf_coef=config.VF_COEF,
        ent_coef=config.ENT_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        verbose=1,
        tensorboard_log=config.log_path,
        device="cpu"
    )
    
    return model


def create_checkpoint_callback(config):
    """Create checkpoint callback for saving model during training"""
    
    callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ,
        save_path=config.save_path,
        name_prefix="ppo_mg_road_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    return callback
