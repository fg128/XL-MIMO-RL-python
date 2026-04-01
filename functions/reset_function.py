import numpy as np

from classes.logged_signals import LoggedSignals
from classes.config import Config
def reset_function(config: Config):
    """Resets the environment to a random initial state.

    Args:
        config: Config object.

    Returns:
        initial_obs:    Initial observation vector (6,) float32.
        logged_signals: LoggedSignals object containing environment state.
    """
    # 1. Randomize Locations
    # bx = (np.random.rand() - 0.5) * 2 * config.max_x
    # bz = 20 + np.random.rand() * (config.max_z - 20)
    # bob_loc = np.array([10, 0.0, 15])

    # ex = bx + (np.random.rand() - 0.5) * 20   # Eve +/-10m near Bob x
    # ez = bz + (np.random.rand() - 0.5) * 20    # Eve +/-10m near Bob z
    # eve_loc = np.array([10, 0.0, 12])

    bob_loc = np.array([-52.4, 0, 44.3])
    eve_loc = np.array([32.7, 0, 13.6])
    bx, _, bz = bob_loc
    ex, _, ez = eve_loc

    logged_signals = LoggedSignals(
        bob_loc=bob_loc,
        eve_loc=eve_loc,
    )

    initial_obs = np.array([
        bx / config.max_x,                  # Absolute Beam Depth
        bz / config.max_z,                  # Absolute Beam Height
        ex / config.max_x,                  # Absolute Eve Depth
        ez / config.max_z,                  # Absolute Eve Height
    ], dtype=np.float32)

    return initial_obs, logged_signals
