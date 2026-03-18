import numpy as np

from classes.config import Config
from functions.get_channel import get_channel
from functions.compute_sr import compute_secrecy_rate_from_W


def mrt_no_an(
    config: Config,
    bob_loc: np.ndarray,
    eve_loc: np.ndarray,
) -> float:
    """Secrecy rate using true Maximum Ratio Transmission with no Artificial Noise.

    Uses the analytic MRT beamformer W = h_bob / ||h_bob||, which maximises
    Bob's received SNR. All power goes to the signal (psf=1.0), injecting zero
    artificial noise. This is the lower bound: it shows that AN genuinely
    improves secrecy.

    Args:
        config:  Config object.
        bob_loc: Bob's location (3,).
        eve_loc: Eve's location (3,).

    Returns:
        secrecy_rate: Secrecy rate in bps/Hz.
    """
    h_bob = get_channel(config, bob_loc)
    h_eve = get_channel(config, eve_loc)

    W_mrt = h_bob / np.linalg.norm(h_bob)  # (Nt, 1), true MRT beamformer

    return compute_secrecy_rate_from_W(config, h_bob, h_eve, W_mrt, 1.0)
