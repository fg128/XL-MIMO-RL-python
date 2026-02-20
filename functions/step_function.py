import numpy as np

from functions.do_action import do_action
from functions.get_channel import get_channel
from classes.config import Config
from classes.logged_signals import LoggedSignals
from functions.visualise import visualise

# Global toggles
_step_count = 0


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
    global _step_count, _verbose_reward
    _step_count += 1

    # 1. Unpack current state
    curr_beam_idx = logged_signals.current_beam_idx
    curr_psf = logged_signals.current_psf
    curr_psf_idx = int(np.argmin(np.abs(config.psf_codebook - curr_psf)))
    bx = logged_signals.bob_loc[0]
    bz = logged_signals.bob_loc[2]
    ex = logged_signals.eve_loc[0]
    ez = logged_signals.eve_loc[2]

    # 2. Execute action
    next_beam_idx, next_psf_idx = do_action(
        action, curr_beam_idx, curr_psf_idx,
        config.size_cb, config.psf_N)
    next_psf = config.psf_codebook[next_psf_idx]

    # 3. Update beam & power splitting factor from codebooks
    W = config.w_beam_codebook[:, next_beam_idx].reshape(-1, 1)  # (Nt, 1)
    psf = config.psf_codebook[next_psf_idx]

    # 4. Power allocation between signal and AN
    P_s = config.P_total_watts * psf
    P_an = config.P_total_watts * (1 - psf)

    # 5. Get channels for Bob and Eve
    h_bob = get_channel(config, logged_signals.bob_loc)  # (Nt, 1)
    h_eve = get_channel(config, logged_signals.eve_loc)  # (Nt, 1)

    Nt = config.Nt

    # 6. Transmitted signal sent
    s = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2) # Transmitted symbol
    z = (np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)) / np.sqrt(2) #  Random noise vector
    V = np.eye(Nt) - (h_bob @ h_bob.conj().T) / (np.linalg.norm(h_bob)**2) # Null space of Bob
    x = np.sqrt(P_s)*W*s + np.sqrt(P_an)*(V @ z) # Transmitted signal

    # 7. Received signals
    sigma_bob = np.sqrt(config.noise_power_watts / 2)
    n_bob = sigma_bob * (np.random.randn() + 1j * np.random.randn())
    y_bob = (h_bob.conj().T @ x).item() + n_bob

    sigma_eve = np.sqrt(config.noise_power_watts / 2)
    n_eve = sigma_eve * (np.random.randn() + 1j * np.random.randn())
    y_eve = (h_eve.conj().T @ x).item() + n_eve

    # 8. Received powers
    rx_pwr_bob = np.abs(y_bob) ** 2
    rx_pwr_eve = np.abs(y_eve) ** 2

    # Analytical signal power
    sig_pwr_bob = P_s * np.abs((h_bob.conj().T @ W).item())**2
    sig_pwr_eve = P_s * np.abs((h_eve.conj().T @ W).item())**2

    # Analytical interference power (AN leakage)
    # Bob: h_bob' * V = 0 by construction, so leakage is 0
    an_leakage_bob = P_an * (np.linalg.norm(h_bob.conj().T @ V) ** 2).item()
    an_leakage_eve = P_an * (np.linalg.norm(h_eve.conj().T @ V) ** 2).item()

    # 9. SINR (Signal to Interference plus Noise Ratio)
    SINR_bob = sig_pwr_bob / (config.noise_power_watts + an_leakage_bob)
    SINR_eve = sig_pwr_eve / (config.noise_power_watts + an_leakage_eve)

    # 10. Secrecy rate
    rate_bob = np.log2(1 + SINR_bob)
    rate_eve = np.log2(1 + SINR_eve)
    secrecy_rate = max(0.0, rate_bob - rate_eve)

    # 11. Distance from beam focal point to Bob and Eve
    current_focus_point = config.beam_focal_locs[next_beam_idx, :]
    dist_to_bob = np.linalg.norm(current_focus_point - logged_signals.bob_loc)
    dist_to_eve = np.linalg.norm(current_focus_point - logged_signals.eve_loc)

    # 12. Reward
    SR = secrecy_rate / 8        # Secrecy rate component
    db = dist_to_bob / 100       # Minimise distance to Bob
    de = dist_to_eve / 300       # Avoid Eve slightly

    close_to_bob_bonus = 0.0
    if db < 0.02:
        close_to_bob_bonus = 0.5
    elif db < 0.1:
        close_to_bob_bonus = 0.20
    elif db < 0.2:
        close_to_bob_bonus = 0.15
    elif db < 0.3:
        close_to_bob_bonus = 0.10

    reward = SR - db + de + close_to_bob_bonus
    is_done = False

    # print(f"[SR={SR:.4f}, db={db:.4f}, de={de:.4f}, bonus={close_to_bob_bonus:.2f}, R={reward:.4f}, psf={next_psf:.2f}]")

    # 13. Update logged signals
    logged_signals.current_beam_idx = next_beam_idx
    logged_signals.current_psf = next_psf

    # 14. Next observation
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
        "dist_to_eve": dist_to_eve
    }

    return next_obs, reward, is_done, logged_signals, info
