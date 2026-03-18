import numpy as np

from classes.config import Config
from functions.get_channel import get_channel
from functions.compute_sr import compute_sr_vectorized

_PSF_GRID = np.linspace(0, 1, 101)  # 0.01-step grid used for the search


def exhaustive_search(
    config: Config,
    bob_loc: np.ndarray,
    eve_loc: np.ndarray,
) -> tuple[float, int, float]:
    """Find the beam and PSF combination that maximises secrecy rate.

    Args:
        config:  Config object.
        bob_loc: Bob's location (3,).
        eve_loc: Eve's location (3,).

    Returns:
        best_sr:       Maximum secrecy rate found (bps/Hz).
        best_beam_idx: Beam codebook index achieving the maximum.
        best_psf:      PSF value (float) achieving the maximum.
    """
    h_bob = get_channel(config, bob_loc)
    h_eve = get_channel(config, eve_loc)

    sr_surface = compute_sr_vectorized(config, h_bob, h_eve, _PSF_GRID)

    flat_idx = np.argmax(sr_surface)
    best_psf_row, best_beam_idx = np.unravel_index(flat_idx, sr_surface.shape)

    return float(sr_surface[best_psf_row, best_beam_idx]), int(best_beam_idx), float(_PSF_GRID[best_psf_row])
