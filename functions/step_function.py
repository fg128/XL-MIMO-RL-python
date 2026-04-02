import sys
import select
import termios
from time import time
import tty
import threading
import numpy as np
from numpy import ndarray

from functions.do_action import do_action
from functions.get_channel import get_channel
from classes.config import Config
from classes.logged_signals import LoggedSignals
from functions.visualise import visualise
from test_sac_pretained import get_sac_output

# Global toggles
_step_count = 0
_verbose = False
_save_checkpoint = False


def start_verbose_toggle():
    """Start a background daemon thread that listens for keypresses:
      'v' — toggle verbose per-step output
      'c' — request a model checkpoint save
    """
    def _listen():
        global _verbose, _save_checkpoint
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch == 'v':
                        _verbose = not _verbose
                        state = "ON" if _verbose else "OFF"
                        print(f"\n[Verbose {state}]\n")
                    elif ch == 'c':
                        _save_checkpoint = True
                        print("\n[Checkpoint requested...]\n")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    t = threading.Thread(target=_listen, daemon=True)
    t.start()


def step_function(action: ndarray, logged_signals: LoggedSignals, config: Config):
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
    global _step_count
    _step_count += 1

    # 1. Execute action
    W, psf = get_sac_output(None, logged_signals.bob_loc, logged_signals.eve_loc, config.vae, config.scaler, action=action)
    W = W.conj()

    # 3. Power allocation between signal and AN
    P_s = config.P_total_watts * psf
    P_an = config.P_total_watts * (1 - psf) 
    
    # 4. Get channels for Bob and Eve (use per-episode NLOS vectors for block fading)
    h_bob = get_channel(config, logged_signals.bob_loc)
    h_eve = get_channel(config, logged_signals.eve_loc)

    Nt = config.Nt

    # 5. Transmitted signal sent
    V = np.eye(Nt) - (h_bob @ h_bob.conj().T) / (np.linalg.norm(h_bob)**2) # Null space of Bob

    # Analytical signal power
    sig_pwr_bob = P_s * np.abs((h_bob.conj().T @ W).item())**2
    sig_pwr_eve = P_s * np.abs((h_eve.conj().T @ W).item())**2

    # Analytical interference power (AN leakage)
    # Bob: h_bob' * V = 0 by construction, so leakage is 0
    an_leakage_bob = P_an * (np.linalg.norm(h_bob.conj().T @ V) ** 2).item()
    an_leakage_eve = P_an * (np.linalg.norm(h_eve.conj().T @ V) ** 2).item()

    # 8. SINR (Signal to Interference plus Noise Ratio)
    # print(f"Signal Power at Bob: {sig_pwr_bob:.4e} W | AN Leakage at Bob: {an_leakage_bob:.4e} W | Noise Power: {config.noise_power_watts:.4e} W")
    SINR_bob = sig_pwr_bob / (config.noise_power_watts + an_leakage_bob)
    SINR_eve = sig_pwr_eve / (config.noise_power_watts + an_leakage_eve)

    # 9. Secrecy rate
    rate_bob = np.log2(1 + SINR_bob)
    rate_eve = np.log2(1 + SINR_eve)
    # print(f"Bob Rate: {rate_bob:.4f} bps/Hz | Eve Rate: {rate_eve:.4f} bps/Hz")

    secrecy_rate = max(0.0, rate_bob - rate_eve)

    # 11. Reward
    reward = secrecy_rate
    # print(f"Secrecy Rate: {secrecy_rate:.4f} bps/Hz")

    is_done = False

    # if _verbose:
        # print(f"[VERBOSE] Reward: {reward:.4f} | PSF: {psf:.3f} | Bob Rate: {rate_bob:.4f} bps/Hz | Eve Rate: {rate_eve:.4f} bps/Hz | Secrecy Rate: {secrecy_rate:.4f} bps/Hz | Bob AN Leakage: {an_leakage_bob:.4e} W | Eve AN Leakage: {an_leakage_eve:.4e} W   | Bob Loc: ({logged_signals.bob_loc[0]:.1f}, {logged_signals.bob_loc[2]:.1f}) | Eve Loc: ({logged_signals.eve_loc[0]:.1f}, {logged_signals.eve_loc[2]:.1f})")

    # 12. Next observation
    bx, bz = logged_signals.bob_loc[0], logged_signals.bob_loc[2]
    ex, ez = logged_signals.eve_loc[0], logged_signals.eve_loc[2]


    if _step_count % config.show_plot_every_nth_steps == 0:
        visualise(W, psf, bx, bz, ex, ez, config, _step_count)

    info = {
        "secrecy_rate": secrecy_rate,
        "bob_loc": logged_signals.bob_loc,
        "eve_loc": logged_signals.eve_loc,
    }

    is_done = True # Each episode is just 1 step to simplify training and focus on learning the optimal beamforming for each scenario

    return None, reward, is_done, logged_signals, info
