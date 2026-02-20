import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from functions.get_channel import get_channel
from classes.config import Config

# Global variable to keep track of the figure
_FIG_AX = None

def visualise(W: ndarray, psf: float, bx: float, bz: float, ex: float, ez: float, config: Config):
    """Visualizes the SNR map with Bob and Eve locations.

    Args:
        W:      Beamforming weight vector (Nt, 1) or (Nt,).
        psf:    Power splitting factor (float).
        bx, bz: Bob's x, z coordinates.
        ex, ez: Eve's x, z coordinates.
        config: Config object.
    """
    global _FIG_AX
    plt.ion() # Turn on interactive mode
    W = np.asarray(W).reshape(-1, 1)

    # 1. Simulate Field Response
    x_range = np.arange(-config.max_x, config.max_x + config.resolution, config.resolution)
    z_range = np.arange(1, config.max_z + config.resolution, config.resolution)
    X_grid, Z_grid = np.meshgrid(x_range, z_range)

    SNR_linear = np.zeros_like(X_grid, dtype=float)

    print(f"Computing SNR for psf={psf:.2f}...")

    # Loop through grid
    for i in range(X_grid.size):
        row, col = np.unravel_index(i, X_grid.shape)
        probe_loc = np.array([X_grid[row, col], 0.0, Z_grid[row, col]])

        h = get_channel(config, probe_loc)
        rx_signal = (h.conj().T @ W).item()
        sig_power = np.abs(rx_signal) ** 2
        SNR_linear[row, col] = sig_power / config.noise_power_watts

    SNR_dB = 10 * np.log10(SNR_linear + 1e-30)
    max_val = np.max(SNR_dB)

    # 2. Visualization
    # Instead of plt.subplots(), check if figure exists
    if _FIG_AX is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        _FIG_AX = (fig, ax)
    else:
        fig, ax = _FIG_AX # type: ignore
        ax.clear() # Clear the old frame
    c = ax.pcolormesh(X_grid, Z_grid, SNR_dB, cmap='jet', vmin=-10, vmax=int(max_val), shading='auto')
    fig.colorbar(c, ax=ax, label='SNR (dB)')

    ax.set_title(f"SNR Distribution - psf={psf:.2f}")
    ax.set_xlabel("Lateral (X) [m]")
    ax.set_ylabel("Depth (Z) [m]")
    ax.set_aspect('equal')

    # Plot Bob (green star)
    if not np.isnan(bx):
        ax.plot(bx, bz, '*', markersize=15, markerfacecolor='g', markeredgecolor='k', label='BOB')
        ax.annotate('  BOB', (bx, bz), color='white', fontweight='bold')

    # Plot Eve (red cross)
    if not np.isnan(ex):
        ax.plot(ex, ez, 'x', markersize=15, linewidth=3, color='r', label='EVE')
        ax.annotate('  EVE', (ex, ez), color='white', fontweight='bold')

    # Plot Antenna Array (red squares at z=0)
    ax.plot(config.pos[0, :], config.pos[2, :], 'rs', markersize=2, label='Antennas')

    ax.legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1) # This allows the GUI to refresh without blocking
