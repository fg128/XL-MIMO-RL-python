import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import os
from datetime import datetime

from functions.get_channel import get_channel
from classes.config import Config


# Global variable to store the directory for the current run
_RUN_DIR = None

def get_run_directory():
    global _RUN_DIR
    if _RUN_DIR is None:
        # Create a unique folder name like: run_20231027_143005
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _RUN_DIR = f"training_results/run_{timestamp}"
        os.makedirs(_RUN_DIR, exist_ok=True)
    return _RUN_DIR


def visualise(W: ndarray, psf: float, bx: float, bz: float, ex: float, ez: float, config: Config, step: int):
    """Visualizes the SNR map with Bob and Eve locations.

    Args:
        W:      Beamforming weight vector (Nt, 1) or (Nt,).
        psf:    Power splitting factor (float).
        bx, bz: Bob's x, z coordinates.
        ex, ez: Eve's x, z coordinates.
        config: Config object.
    """
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
    fig, ax = plt.subplots(figsize=(12, 8)) # Slightly wider for the sidebar info
    c = ax.pcolormesh(X_grid, Z_grid, SNR_dB, cmap='jet', vmin=-10, vmax=int(max_val), shading='auto')
    fig.colorbar(c, ax=ax, label='SNR (dB)')

    # Add settings metadata as text on the right side of the plot
    info_text = (
        f"Step: {step}\n"
        f"PSF: {psf:.3f}\n"
        f"Nt: {config.Nt}\n"
        f"P_total: {config.P_total_watts}W\n"
        f"Bob Loc: ({bx:.1f}, {bz:.1f})\n"
        f"Eve Loc: ({ex:.1f}, {ez:.1f})"
    )
    plt.gcf().text(0.85, 0.5, info_text, fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_title(f"SNR Map | Step {step} | PSF {psf:.2f}")
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

    # 3. Save with metadata in the filename
    run_dir = get_run_directory()
    # Filename includes step and PSF for quick sorting in folders
    file_name = f"step_{step:06d}_psf_{psf:.2f}.png"
    save_path = os.path.join(run_dir, file_name)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # Also save the config as a text file in the same folder (once per run)
    config_log = os.path.join(run_dir, "settings.txt")
    if not os.path.exists(config_log):
        with open(config_log, "w") as f:
            f.write(f"Run started at: {datetime.now()}\n")
            f.write(f"Antennas: {config.Nt}\n")
            f.write(f"Resolution: {config.resolution}\n")

    print(f"Plot saved to {save_path}")
