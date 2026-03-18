import numpy as np

from classes.config import Config
from functions.get_channel import get_channel
from functions.compute_sr import compute_secrecy_rate_from_W


def fixed_power_allocation(
    config: Config,
    bob_loc: np.ndarray,
    eve_loc: np.ndarray,
    psf_values: tuple = (0.5, 0.8),
) -> dict[float, float]:
    """Secrecy rate using true MRT beamformer with fixed power allocation.

    Uses the analytic MRT beamformer W = h_bob / ||h_bob|| and evaluates the
    secrecy rate at each requested (fixed) PSF value.

    Args:
        config:     Config object.
        bob_loc:    Bob's location (3,).
        eve_loc:    Eve's location (3,).
        psf_values: PSF values to evaluate.

    Returns:
        results: Dict mapping each psf_value to its secrecy rate (bps/Hz).
    """
    h_bob = get_channel(config, bob_loc)
    h_eve = get_channel(config, eve_loc)

    W_mrt = h_bob / np.linalg.norm(h_bob)  # (Nt, 1), true MRT beamformer

    results = {}
    for psf_val in psf_values:
        sr = compute_secrecy_rate_from_W(config, h_bob, h_eve, W_mrt, float(psf_val))
        results[psf_val] = sr

    return results
