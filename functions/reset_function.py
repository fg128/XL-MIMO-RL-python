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
    bx = (np.random.rand() - 0.5) * 2 * config.max_x
    bz = 20 + np.random.rand() * (config.max_z - 20)
    bob_loc = np.array([bx, 0.0, bz])

    ex = bx + (np.random.rand() - 0.5) * 20   # Eve +/-10m near Bob x
    ez = bz + (np.random.rand() - 0.5) * 20    # Eve +/-10m near Bob z
    eve_loc = np.array([ex, 0.0, ez])

    # 2. Initialize agent randomly
    start_beam_idx = np.random.randint(0, config.size_cb)
    cx, _, cz = config.beam_focal_locs[start_beam_idx, :]
    r_beam = np.sqrt(cx**2 + cz**2)
    theta_beam = np.arctan2(cx, cz)
    start_psf = np.random.rand()

    logged_signals = LoggedSignals(
        bob_loc=bob_loc,
        eve_loc=eve_loc,
        ideal_r=r_beam,
        ideal_theta=theta_beam,
        ideal_psf=start_psf,
    )

    # Absolute polar coordinates
    r_bob = np.sqrt(bx**2 + bz**2)
    theta_bob = np.arctan2(bx, bz)

    r_eve = np.sqrt(ex**2 + ez**2)
    theta_eve = np.arctan2(ex, ez)

    # Calculate polar deltas (how far is beam from Bob/Eve?)
    delta_r_bob = r_bob - r_beam
    delta_theta_bob = theta_bob - theta_beam

    delta_r_eve = r_eve - r_beam
    delta_theta_eve = theta_eve - theta_beam

    # Normalization constants for stabilty
    max_r = np.sqrt(config.max_x**2 + config.max_z**2)
    max_theta_sweep = np.pi / 2 # ~90 degrees the delta spread

    initial_obs = np.array([
        r_beam / max_r,                  # Absolute Beam Depth
        theta_beam / max_theta_sweep,    # Absolute Beam Angle
        start_psf,                        # Current Power Split
        delta_r_bob / max_r,            # Distance from Bob
        delta_theta_bob / max_theta_sweep,
        delta_r_eve / max_r,
        delta_theta_eve / max_theta_sweep,
    ], dtype=np.float32)

    return initial_obs, logged_signals
