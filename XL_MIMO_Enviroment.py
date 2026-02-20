import os
import sys
from classes.logged_signals import LoggedSignals
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Ensure the python/ directory is on the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classes.config import Config
from functions.step_function import step_function
from functions.reset_function import reset_function


class XLMIMOEnv(gym.Env):
    """XL-MIMO Beamforming Gymnasium Environment.

    Observation (6,):
        [beam_idx_norm, psf, delta_bob_x, delta_bob_z, delta_eve_x, delta_eve_z]

    Actions (Discrete 11):
        0=Stay, 1=Angle+1, 2=Angle+8, 3=Angle-1, 4=Angle-8,
        5=Range+1, 6=Range+5, 7=Range-1, 8=Range-5,
        9=PSF+1, 10=PSF-1
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config=None):
        super().__init__()

        self.config = config if config is not None else Config(yaml_path='config.yaml')

        # Observation space: [beam_focal_r, beam_focal_theta, psf, delta_bob_r, delta_bob_theta, delta_eve_r, delta_eve_theta]
        self.observation_space = spaces.Box(
            low=np.array([-1, 0, 0, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([+1, +1, +1, +1, +1, +1, +1], dtype=np.float32),
            dtype=np.float32,
        )

        # Action space : [delta_ideal_r, delta_ideal_theta, delta_psf]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.logged_signals: LoggedSignals  = None # type: ignore

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        initial_obs, self.logged_signals = reset_function(self.config)
        return initial_obs, {}

    def step(self, action):
        next_obs, reward, is_done, self.logged_signals, info = step_function(
            action, self.logged_signals, self.config
        )
        truncated = False
        return next_obs, float(reward), bool(is_done), truncated, info
