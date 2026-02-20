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
    b_x = (np.random.rand() - 0.5) * 2 * config.max_x
    b_z = 20 + np.random.rand() * (config.max_z - 20)
    bob_loc = np.array([b_x, 0.0, b_z])

    e_x = b_x + (np.random.rand() - 0.5) * 20   # Eve +/-10m near Bob x
    e_z = b_z + (np.random.rand() - 0.5) * 20    # Eve +/-10m near Bob z
    eve_loc = np.array([e_x, 0.0, e_z])

    # 2. Initialize agent randomly
    start_beam_idx = np.random.randint(0, config.size_cb)
    start_psf_idx = np.random.randint(0, len(config.psf_codebook))
    start_psf = config.psf_codebook[start_psf_idx]

    logged_signals = LoggedSignals(
        bob_loc=bob_loc,
        eve_loc=eve_loc,
        current_beam_idx=start_beam_idx,
        current_psf=start_psf,
    )

    # 3. Set initial observations
    current_focus_point = config.beam_focal_locs[start_beam_idx, :]
    delta_bob_x = (logged_signals.bob_loc[0] - current_focus_point[0]) / (2 * config.max_x)
    delta_bob_z = (logged_signals.bob_loc[2] - current_focus_point[2]) / config.max_z
    delta_eve_x = (logged_signals.eve_loc[0] - current_focus_point[0]) / (2 * config.max_x)
    delta_eve_z = (logged_signals.eve_loc[2] - current_focus_point[2]) / config.max_z

    initial_obs = np.array([
        start_beam_idx / config.size_cb,
        start_psf,
        delta_bob_x,
        delta_bob_z,
        delta_eve_x,
        delta_eve_z,
    ], dtype=np.float32)

    return initial_obs, logged_signals
