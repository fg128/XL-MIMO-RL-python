from classes.config import Config
from classes.logged_signals import LoggedSignals

import numpy as np
from numpy import ndarray


def do_action(action: ndarray, logged_signals: LoggedSignals, config: Config):
    """Routes the DQN action decision to performing the action.

    Args:
        action:        Action chosen by DQN (0=Stay, 1-8=Move beam, 9-10=Power).
        curr_beam_idx: Current beamforming codebook index (0-based).
        curr_psf_idx:  Current power splitting codebook index (0-based).
        size_cb:       Total size of the codebook (e.g., 1024).
        psf_N:         N = psf_codebook_size - 1.

    Returns:
        next_beam_idx: New beamforming codebook index (0-based).
        next_psf_idx:  New power splitting codebook index (0-based).
    """
    # 1. Scale the continuous actions to physical step sizes
    delta_r = action[0] * config.max_r_step
    delta_theta = action[1] * config.max_theta_step
    delta_psf = action[2] * config.max_psf_step

    # 2. Update the IDEAL continuous state (Internal to the agent)
    max_r = np.sqrt(config.max_x**2 + config.max_z**2)

    # Clip all so not out of bounds
    ideal_r = np.clip(logged_signals.ideal_r + delta_r, 0.1, max_r)
    ideal_theta = np.clip(logged_signals.ideal_theta + delta_theta, -np.pi/2, np.pi/2)
    ideal_psf = np.clip(logged_signals.ideal_psf + delta_psf, 0.0, 1.0)

    return ideal_r, ideal_theta, ideal_psf


