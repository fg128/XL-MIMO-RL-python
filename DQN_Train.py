import os

import torch

from classes.custom_metrics_callback import CustomMetricsCallback
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import sys
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit

# Ensure the python/ directory is on the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classes.config import Config
from XL_MIMO_Enviroment import XLMIMOEnv
from functions.step_function import start_verbose_toggle
torch.set_num_threads(4)

def make_env(config_obj: Config):
    def _init():
        base_env = XLMIMOEnv(config=config_obj)
        return TimeLimit(
            base_env, 
            max_episode_steps=config_obj.max_steps_per_episode
        )
    return _init

# -------------------------------------------------------------------------
# 2. DEFINE OR LOAD DQN AGENT
# -------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config()

    # We pass the list of FUNCTIONS returned by make_env, not the environments themselves
    env = SubprocVecEnv([make_env(config) for _ in range(config.num_cores)])
    log_dir = "./logs/"

    if config.sac_model_path and os.path.isfile(config.sac_model_path):
        print(f'Loading SAC agent from {config.sac_model_path}...')
        sac_model = SAC.load(
            config.sac_model_path, 
            env=env,
            device='cpu',
            tensorboard_log=log_dir,
            
            )
    else:
        print('No existing model found. Creating new DQN Agent...')

        sac_model = SAC(
            'MlpPolicy',
            env,
            # --- Network Architecture: 256-256-256 hidden layers ---
            policy_kwargs=dict(net_arch=dict(
                pi=[256, 256, 256],
                qf=[256, 256, 256]
            )),

            # --- Optimizer ---
            learning_rate=1e-4,
            gamma=0,                     # DiscountFactor
            tau=1e-3,                       # TargetSmoothFactor (soft update)
            target_update_interval=1,       # Soft update every step
            batch_size=256,                 # MiniBatchSize
            buffer_size=100_000,            # ExperienceBufferLength
            learning_starts=256,            # Start training after filling one batch

            # --- General ---
            train_freq=1,
            gradient_steps=1,
            verbose=0,
            tensorboard_log=log_dir,
            device='cpu',
        )

    # -------------------------------------------------------------------------
    # 3. TRAINING LOOP
    # -------------------------------------------------------------------------
    # Instantiate the callback to log secrecy rate metrics to tensorboard
    metrics_callback = CustomMetricsCallback(config=config)
    # start_verbose_toggle()
    print("Press 'v' during training to toggle verbose output. Press 'c' to save a checkpoint.")
    print('Starting Training...')

    # --- TEMPORARY DEBUGGING BLOCK START ---
    # 1. Force the model to skip the completely random action warmup phase
    sac_model.learning_starts = 0 
    
    # 2. Intercept the policy's predict method to ALWAYS force deterministic=True
    original_predict = sac_model.policy.predict
    sac_model.policy.predict = lambda obs, state=None, episode_start=None, deterministic=False: original_predict(obs, state, episode_start, deterministic=True)
    # --- TEMPORARY DEBUGGING BLOCK END ---

    sac_model.learn(
        total_timesteps=config.total_timesteps,
        callback=metrics_callback,
    )
    print('Training Complete.')


    env.close()

    # -------------------------------------------------------------------------
    # 4. SAVE AGENT
    # -------------------------------------------------------------------------
    # save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.save_file_path)
    # sac_model.save(save_path)
    # print(f'Agent saved to {save_path}')
