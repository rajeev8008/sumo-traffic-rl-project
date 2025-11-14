"""
PPO Agent for SUMO Traffic Signal Control
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    BaseCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
import os
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field, asdict
from stable_baselines3.common.vec_env import VecEnv 


class OverfittingMonitorCallback(BaseCallback):
    """
    Custom callback to monitor training vs validation performance.
    """
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.check_freq = check_freq
        self.train_rewards = []
        self.last_mean_reward = None
        
    def _on_step(self) -> bool:
        """Called after each environment step."""
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                recent_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                mean_reward = np.mean(recent_rewards)
                self.train_rewards.append(mean_reward)
                
                if self.verbose > 0:
                    print(f"Step {self.num_timesteps}: Mean Episode Reward = {mean_reward:.3f}")
        
        return True


@dataclass
class PPOAgentConfig:
    """
    Configuration for PPO agent training with optimized parameters.
    """
    
    # Network architecture
    NET_ARCH: List[int] = field(default_factory=lambda: [256, 256, 128])
    ACTIVATION_FN: Callable = nn.ReLU
    
    # PPO hyperparameters - OPTIMIZED for traffic control
    LEARNING_RATE: float = 5e-5          
    N_STEPS: int = 4096                  
    BATCH_SIZE: int = 256                
    N_EPOCHS: int = 10
    GAMMA: float = 0.995                 
    GAE_LAMBDA: float = 0.95
    
    # Regularization
    ENT_COEF: float = 0.005              
    VF_COEF: float = 0.25                
    MAX_GRAD_NORM: float = 0.5
    
    # Training stability
    CLIP_RANGE: float = 0.2
    CLIP_RANGE_VF: Optional[float] = None
    
    # Policy network arguments property
    @property
    def POLICY_KWARGS(self) -> Dict[str, Any]:
        return {
            "net_arch": self.NET_ARCH,
            "activation_fn": self.ACTIVATION_FN,
            "ortho_init": True,
            "log_std_init": -1.0, 
        }
    
    def to_dict(self):
        """Return config as dictionary for PPO initialization."""
        d = asdict(self)
        d['policy_kwargs'] = self.POLICY_KWARGS
        
        if 'NET_ARCH' in d: del d['NET_ARCH']
        if 'ACTIVATION_FN' in d: del d['ACTIVATION_FN']
        
        return {k.lower(): v for k, v in d.items()}


def create_ppo_agent(
    env_creator: VecEnv, 
    model_name: str = "ppo_agent_mg_road",
    seed: int = 42,
    verbose: int = 1,
    use_config: PPOAgentConfig = PPOAgentConfig(),
) -> PPO:
    """
    Create a PPO agent for traffic signal control.
    """
    config_dict = use_config.to_dict()
    
    agent = PPO(
        policy="MlpPolicy",
        env=env_creator, 
        **config_dict,
        verbose=verbose,
        device="cpu",
        seed=seed,
        tensorboard_log="./logs/ppo_mg_road",
    )
    
    return agent


def setup_evaluation_callbacks(
    eval_env: VecEnv, 
    eval_freq: int = 5000,
    n_eval_episodes: int = 5,
    best_model_save_path: str = "./models/",
    model_name: str = "ppo_agent_mg_road",
    patience: int = 3,
) -> tuple[EvalCallback, OverfittingMonitorCallback]:
    """
    Setup callbacks for monitoring training and early stopping.
    """
    os.makedirs(best_model_save_path, exist_ok=True)
    
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=patience,
        verbose=1,
    )
    
    eval_callback = EvalCallback(
        eval_env=eval_env, 
        callback_after_eval=stop_train_callback,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=best_model_save_path,
        log_path="./logs/eval_mg_road",
        deterministic=True,
        verbose=1,
    )
    
    overfitting_monitor = OverfittingMonitorCallback(check_freq=1000, verbose=1)
    
    return eval_callback, overfitting_monitor


def load_ppo_agent(model_path: str, env=None) -> PPO:
    """
    Load a trained PPO agent from disk.
    """
    agent = PPO.load(model_path, env=env)
    return agent


def get_action(agent: PPO, observation: np.ndarray, deterministic: bool = False) -> int:
    """
    Get action from agent for a given observation.
    """
    action, _ = agent.predict(observation, deterministic=deterministic)
    return int(action)


if __name__ == "__main__":
    print("PPO Agent Configuration for Traffic Signal Control")
    print("=" * 60)
    config = PPOAgentConfig()
    for key, value in config.to_dict().items():
        print(f"{key}: {value}")