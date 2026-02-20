import os

from classes.custom_metrics_callback import CustomMetricsCallback
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import sys
import numpy as np
from stable_baselines3 import DQN
from gymnasium.wrappers import TimeLimit

# Ensure the python/ directory is on the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classes.config import Config
from XL_MIMO_Enviroment import XLMIMOEnv


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
model_file = ''                    # Set to path of existing .zip model to resume
save_file = 'trained_agent_3.zip'  # Where to save the trained model

# -------------------------------------------------------------------------
# 1. DEFINE REINFORCEMENT LEARNING ENVIRONMENT
# -------------------------------------------------------------------------
config = Config(yaml_path='config.yaml')
env = TimeLimit(XLMIMOEnv(config=config), max_episode_steps=config.max_steps_per_episode)

# -------------------------------------------------------------------------
# 2. DEFINE OR LOAD DQN AGENT
# -------------------------------------------------------------------------
total_timesteps = config.max_episodes * config.max_steps_per_episode  # 50,000

if model_file and os.path.isfile(model_file):
    print(f'Loading existing agent from {model_file}...')
    model = DQN.load(model_file, env=env)
    # Adjust epsilon for resuming training
    model.exploration_rate = 0.5

else:
    print('No existing model found. Creating new DQN Agent...')

    # Epsilon decay: MATLAB EpsilonDecay=1e-4 per step
    # ~9500 steps to go from 1.0 to 0.05
    exploration_fraction = 9500 / total_timesteps
    log_dir = "./logs/"

    model = DQN(
        'MlpPolicy',
        env,
        # --- Network Architecture: 256-256 hidden layers ---
        policy_kwargs=dict(net_arch=[256, 256]),

        # --- Optimizer ---
        learning_rate=1e-4,
        max_grad_norm=1.0,              # GradientThreshold

        # --- DQN Hyperparameters ---
        gamma=0.99,                     # DiscountFactor
        tau=1e-3,                       # TargetSmoothFactor (soft update)
        target_update_interval=1,       # Soft update every step
        batch_size=256,                 # MiniBatchSize
        buffer_size=100_000,            # ExperienceBufferLength
        learning_starts=256,            # Start training after filling one batch

        # --- Exploration Strategy ---
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=exploration_fraction,

        # --- General ---
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=log_dir,
        device='cpu',
    )

# -------------------------------------------------------------------------
# 3. TRAINING LOOP
# -------------------------------------------------------------------------
# Instantiate the callback to log secrecy rate metrics to tensorboard
metrics_callback = CustomMetricsCallback()

print('Starting Training...')
model.learn(
    total_timesteps=total_timesteps,
    callback=metrics_callback
)
print('Training Complete.')

# -------------------------------------------------------------------------
# 4. SAVE AGENT
# -------------------------------------------------------------------------
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file)
model.save(save_path)
print(f'Agent saved to {save_path}')
