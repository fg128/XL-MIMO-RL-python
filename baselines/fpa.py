import numpy as np
from classes.config import Config
from functions.get_channel import get_channel


def compute_secrecy_rate(config: Config, bob_loc, eve_loc, phi):
    h_bob = get_channel(config, bob_loc)
    h_eve = get_channel(config, eve_loc)
    Nt = config.Nt

    # MRT direction with fixed power split

    W = h_bob / np.linalg.norm(h_bob)
    P_s  = config.P_total_watts * phi
    P_an = config.P_total_watts * (1 - phi)

    # Null space of Bob's channel for artificial noise
    V = np.eye(Nt) - (h_bob @ h_bob.conj().T) / (np.linalg.norm(h_bob) ** 2)

    sig_pwr_bob = P_s * np.abs((h_bob.conj().T @ W).item()) ** 2
    sig_pwr_eve = P_s * np.abs((h_eve.conj().T @ W).item()) ** 2
    # print(f"DEBUG: sig_pwr_bob={sig_pwr_bob}, sig_pwr_eve={sig_pwr_eve}")

    # Bob: h_bob' @ V = 0 by construction, so AN leakage on Bob is zero
    an_leakage_bob = P_an * float(np.linalg.norm(h_bob.conj().T @ V) ** 2)
    an_leakage_eve = P_an * float(np.linalg.norm(h_eve.conj().T @ V) ** 2)

    SINR_bob = sig_pwr_bob / (config.noise_power_watts + an_leakage_bob)
    SINR_eve = sig_pwr_eve / (config.noise_power_watts + an_leakage_eve)

    rate_bob = np.log2(1 + SINR_bob)
    rate_eve = np.log2(1 + SINR_eve)
    return max(0.0, float(rate_bob - rate_eve))
