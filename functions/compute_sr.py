from typing import Union

import numpy as np

from classes.config import Config


def compute_secrecy_rate(
    config: Config,
    h_bob: np.ndarray,
    h_eve: np.ndarray,
    beam_idx: int,
    psf: float,
) -> float:
    """Compute the secrecy rate for a given beam and PSF selection.

    Args:
        config:   Config object.
        h_bob:    Channel vector to Bob (Nt, 1) complex.
        h_eve:    Channel vector to Eve (Nt, 1) complex.
        beam_idx: Index into the beam codebook.
        psf:   Power splitting factor (0.0 to 1.0) for the signal; remainder is AN.

    Returns:
        secrecy_rate: Scalar secrecy rate in bps/Hz.
    """
    W = config.w_beam_codebook[:, beam_idx].reshape(-1, 1)  # (Nt, 1)

    # Power allocation between signal and AN
    P_s = config.P_total_watts * psf
    P_an = config.P_total_watts * (1 - psf)

    Nt = config.Nt

    # Transmitted signal sent
    s = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)  # Transmitted symbol
    z = (np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)) / np.sqrt(2)  # Random noise vector
    V = np.eye(Nt) - (h_bob @ h_bob.conj().T) / (np.linalg.norm(h_bob)**2)  # Null space of Bob
    x = np.sqrt(P_s) * W * s + np.sqrt(P_an) * (V @ z)  # Transmitted signal

    # Received signals
    sigma_bob = np.sqrt(config.noise_power_watts / 2)
    n_bob = sigma_bob * (np.random.randn() + 1j * np.random.randn())
    y_bob = (h_bob.conj().T @ x).item() + n_bob

    sigma_eve = np.sqrt(config.noise_power_watts / 2)
    n_eve = sigma_eve * (np.random.randn() + 1j * np.random.randn())
    y_eve = (h_eve.conj().T @ x).item() + n_eve

    # Received powers
    rx_pwr_bob = np.abs(y_bob) ** 2
    rx_pwr_eve = np.abs(y_eve) ** 2

    # Analytical signal power
    sig_pwr_bob = P_s * np.abs((h_bob.conj().T @ W).item())**2
    sig_pwr_eve = P_s * np.abs((h_eve.conj().T @ W).item())**2

    # Analytical interference power (AN leakage)
    # Bob: h_bob' * V = 0 by construction, so leakage is 0
    an_leakage_bob = P_an * (np.linalg.norm(h_bob.conj().T @ V) ** 2).item()
    an_leakage_eve = P_an * (np.linalg.norm(h_eve.conj().T @ V) ** 2).item()

    # SINR (Signal to Interference plus Noise Ratio)
    SINR_bob = sig_pwr_bob / (config.noise_power_watts + an_leakage_bob)
    SINR_eve = sig_pwr_eve / (config.noise_power_watts + an_leakage_eve)

    # Secrecy rate
    rate_bob = np.log2(1 + SINR_bob)
    rate_eve = np.log2(1 + SINR_eve)
    secrecy_rate = max(0.0, rate_bob - rate_eve)

    return secrecy_rate


def compute_secrecy_rate_from_W(
    config: Config,
    h_bob: np.ndarray,
    h_eve: np.ndarray,
    W: np.ndarray,
    psf: float,
) -> float:
    """Compute secrecy rate for an arbitrary beamformer W (not restricted to codebook).

    Identical physics to compute_secrecy_rate but accepts W directly instead of
    a codebook beam_idx. Intended for true-MRT and other analytic beamformers.

    Args:
        config:  Config object.
        h_bob:   Channel vector to Bob (Nt, 1) complex.
        h_eve:   Channel vector to Eve (Nt, 1) complex.
        W:       Beamforming vector (Nt, 1) complex, normalised externally.
        psf:     Power splitting factor (0.0 to 1.0) for the signal.

    Returns:
        secrecy_rate: Scalar secrecy rate in bps/Hz.
    """
    P_s = config.P_total_watts * psf
    P_an = config.P_total_watts * (1 - psf)

    Nt = config.Nt

    s = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    z = (np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)) / np.sqrt(2)
    V = np.eye(Nt) - (h_bob @ h_bob.conj().T) / (np.linalg.norm(h_bob)**2)
    x = np.sqrt(P_s) * W * s + np.sqrt(P_an) * (V @ z)

    sigma_bob = np.sqrt(config.noise_power_watts / 2)
    n_bob = sigma_bob * (np.random.randn() + 1j * np.random.randn())
    y_bob = (h_bob.conj().T @ x).item() + n_bob

    sigma_eve = np.sqrt(config.noise_power_watts / 2)
    n_eve = sigma_eve * (np.random.randn() + 1j * np.random.randn())
    y_eve = (h_eve.conj().T @ x).item() + n_eve

    rx_pwr_bob = np.abs(y_bob) ** 2
    rx_pwr_eve = np.abs(y_eve) ** 2

    sig_pwr_bob = P_s * np.abs((h_bob.conj().T @ W).item())**2
    sig_pwr_eve = P_s * np.abs((h_eve.conj().T @ W).item())**2

    an_leakage_bob = P_an * (np.linalg.norm(h_bob.conj().T @ V) ** 2).item()
    an_leakage_eve = P_an * (np.linalg.norm(h_eve.conj().T @ V) ** 2).item()

    SINR_bob = sig_pwr_bob / (config.noise_power_watts + an_leakage_bob)
    SINR_eve = sig_pwr_eve / (config.noise_power_watts + an_leakage_eve)

    rate_bob = np.log2(1 + SINR_bob)
    rate_eve = np.log2(1 + SINR_eve)
    secrecy_rate = max(0.0, rate_bob - rate_eve)

    return secrecy_rate


def compute_sr_vectorized(
    config: Config,
    h_bob: np.ndarray,
    h_eve: np.ndarray,
    psf_values: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """Compute secrecy rate for every (psf, beam) combination via broadcasting.

    No V matrix is ever materialised. Uses the identity:
        ||h_eve.H @ V||^2 = ||h_eve||^2 - |h_eve.H @ h_bob|^2 / ||h_bob||^2

    Args:
        config:     Config object.
        h_bob:      Channel vector to Bob (Nt, 1) complex.
        h_eve:      Channel vector to Eve (Nt, 1) complex.
        psf_values: PSF grid to search (default: linspace(0, 1, 21), i.e. 0.05 steps).

    Returns:
        sr_surface: Array of shape (n_psf, size_cb) with secrecy rates in bps/Hz.
    """
    if psf_values is None:
        psf_values = np.linspace(0, 1, 21)  # 0.05-step grid

    W_all = config.w_beam_codebook  # (Nt, size_cb)

    # Beam gains for all beams in one matmul each
    beam_gains_bob = np.abs(h_bob.conj().T @ W_all).ravel() ** 2  # (size_cb,)
    beam_gains_eve = np.abs(h_eve.conj().T @ W_all).ravel() ** 2  # (size_cb,)

    # AN leakage scalar — avoids the 1024x1024 V matrix
    h_bob_norm_sq = np.linalg.norm(h_bob) ** 2
    cross = (h_eve.conj().T @ h_bob).item()
    h_eve_V_norm_sq = np.linalg.norm(h_eve) ** 2 - abs(cross) ** 2 / h_bob_norm_sq

    P_s  = config.P_total_watts * psf_values        # (n_psf,)
    P_an = config.P_total_watts * (1 - psf_values)  # (n_psf,)

    # Broadcast to (n_psf, size_cb)
    SINR_bob = (P_s[:, None] * beam_gains_bob[None, :]) / config.noise_power_watts
    SINR_eve = (P_s[:, None] * beam_gains_eve[None, :]) / (
        config.noise_power_watts + P_an[:, None] * h_eve_V_norm_sq
    )

    return np.maximum(0.0, np.log2(1 + SINR_bob) - np.log2(1 + SINR_eve))
