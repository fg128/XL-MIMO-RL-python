import numpy as np
from classes.config import Config
from functions.get_channel import get_channel


def compute_secrecy_rate(config: Config, bob_loc, eve_loc):
    h_bob = get_channel(config, bob_loc)
    h_eve = get_channel(config, eve_loc)

    # MRT: beamform directly toward Bob, all power to signal
    W = h_bob / np.linalg.norm(h_bob)
    P_s = config.P_total_watts

    sig_pwr_bob = P_s * np.abs((h_bob.conj().T @ W).item()) ** 2
    sig_pwr_eve = P_s * np.abs((h_eve.conj().T @ W).item()) ** 2

    SINR_bob = sig_pwr_bob / config.noise_power_watts
    SINR_eve = sig_pwr_eve / config.noise_power_watts

    rate_bob = np.log2(1 + SINR_bob)
    rate_eve = np.log2(1 + SINR_eve)
    return max(0.0, float(rate_bob - rate_eve))
