import numpy as np

from functions.do_action import do_action
from functions.get_channel import get_channel
from functions.compute_sr import compute_secrecy_rate
from classes.config import Config
from classes.logged_signals import LoggedSignals
from functions.visualise import visualise

# Global toggles
_step_count = 0
_verbose = False


def step_function(action: int, logged_signals: LoggedSignals, config: Config):
    """Executes one step of the environment.

    Args:
        action:         Integer action (0-10).
        logged_signals: LoggedSignals object containing current environment state.
        config:         Config object.

    Returns:
        next_obs:       Next observation vector (6,) float32.
        reward:         Scalar reward.
        is_done:        Whether the episode is terminated.
        logged_signals: Updated LoggedSignals object.
    """
    global _step_count, _verbose
    _step_count += 1

    # 1. Unpack current state
    curr_beam_idx = logged_signals.current_beam_idx
    curr_psf = logged_signals.current_psf
    bx = logged_signals.bob_loc[0]
    bz = logged_signals.bob_loc[2]
    ex = logged_signals.eve_loc[0]
    ez = logged_signals.eve_loc[2]

    # 2. Execute action
    next_beam_idx, next_psf = do_action(
        action, curr_beam_idx, curr_psf,
        config.size_cb)

    # 3. Update beam & power splitting factor from codebooks
    W = config.w_beam_codebook[:, next_beam_idx].reshape(-1, 1)  # (Nt, 1)

    # 4. Get channels for Bob and Eve
    h_bob = get_channel(config, logged_signals.bob_loc)  # (Nt, 1)
    h_eve = get_channel(config, logged_signals.eve_loc)  # (Nt, 1)

    # 5. Compute secrecy rate
    secrecy_rate = compute_secrecy_rate(config, h_bob, h_eve, next_beam_idx, next_psf)

    # 6. Distance from beam focal point to Bob and Eve
    current_focus_point = config.beam_focal_locs[next_beam_idx, :]
    dist_to_bob = np.linalg.norm(current_focus_point - logged_signals.bob_loc)
    dist_to_eve = np.linalg.norm(current_focus_point - logged_signals.eve_loc)

    # 7. Reward
    SR = secrecy_rate        # Secrecy rate component
    db = dist_to_bob / 100       # Minimise distance to Bob
    de = dist_to_eve / 100       # Avoid Eve slightly

    reward = SR
    is_done = False

    if _verbose:
        print(f"[SR={SR:.4f}, db={db:.4f}, de={de:.4f}, R={reward:.4f}, psf={next_psf:.2f}]")

    # 8. Update logged signals
    logged_signals.current_beam_idx = next_beam_idx
    logged_signals.current_psf = next_psf

    # 9. Next observation
    bx, bz = logged_signals.bob_loc[0], logged_signals.bob_loc[2]
    ex, ez = logged_signals.eve_loc[0], logged_signals.eve_loc[2]
    cx, cz = current_focus_point[0], current_focus_point[2]

    # Absolute polar coordinates
    r_beam = np.sqrt(cx**2 + cz**2)
    theta_beam = np.arctan2(cx, cz)

    r_bob = np.sqrt(bx**2 + bz**2)
    theta_bob = np.arctan2(bx, bz)

    r_eve = np.sqrt(ex**2 + ez**2)
    theta_eve = np.arctan2(ex, ez)

    # Calculate polar deltas (How far is beam from Bob/Eve?)
    delta_r_bob = r_bob - r_beam
    delta_theta_bob = theta_bob - theta_beam

    delta_r_eve = r_eve - r_beam
    delta_theta_eve = theta_eve - theta_beam

    # Normalization constants for stabilty
    max_r = np.sqrt(config.max_x**2 + config.max_z**2)
    max_theta_sweep = np.pi / 2 # ~90 degrees is plenty for the delta spread

    next_obs = np.array([
        r_beam / max_r,                  # Absolute Beam Depth
        theta_beam / max_theta_sweep,    # Absolute Beam Angle
        next_psf,                        # Current Power Split
        delta_r_bob / max_r,
        delta_theta_bob / max_theta_sweep,
        delta_r_eve / max_r,
        delta_theta_eve / max_theta_sweep,
    ], dtype=np.float32)

    if _step_count % config.show_plot_every_nth_steps == 0:
        visualise(W, next_psf, bx, bz, ex, ez, config, _step_count)
        print(f"PSF: {next_psf}, Beam IDX: {next_beam_idx}")
        print(f"Focus Point: {current_focus_point}")

    info = {
        "secrecy_rate": secrecy_rate,
        "dist_to_bob": dist_to_bob,
        "dist_to_eve": dist_to_eve,
        "bob_loc": logged_signals.bob_loc,
        "eve_loc": logged_signals.eve_loc,
    }

    return next_obs, reward, is_done, logged_signals, info
