import numpy as np
from functions.get_config import get_config

# ---------------------------------------------------------------------------
# Demo: Replicate XL_MIMO_Enviroment.m visualization when run directly
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from functions.codebook import codebook
    from functions.move_beam import move_beam
    from functions.visualise import visualise

    config = get_config()

    # Test focal point adjustments (0-based: MATLAB index 300 -> Python index 299)
    current_idx = 299
    right = move_beam(current_idx, 'angle', +1, config.size_cb)
    left = move_beam(current_idx, 'angle', -1, config.size_cb)
    left = move_beam(left, 'range', 5, config.size_cb)

    beam_indexes = [left, current_idx]

    # Form beamforming weights W
    W = np.zeros((config.Nt, 1), dtype=complex)
    print("Targets selected from Codebook:")
    for i in beam_indexes:
        W += config.w_beam_codebook[:, i].reshape(-1, 1)
        coords = config.beam_focal_locs[i, :]
        print(f"  Index {i}: X={coords[0]:.1f} m, Z={coords[2]:.1f} m")

    # Normalise weights
    W = W / np.linalg.norm(W)

    visualise(W, 0.5, 10, 10, 13, 13, config)
