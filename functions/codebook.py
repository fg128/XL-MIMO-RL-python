import numpy as np
from numpy import ndarray


def codebook(Nt: int, pos: ndarray, k: float, size_cb: int, max_x: float, max_z: float):
    """Generates Near-Field beamforming weights in a codebook.

    Args:
        Nt:      Number of antennas.
        pos:     Antenna positions (3 x Nt array).
        k:       Wavenumber (2*pi/lambda).
        size_cb: Total desired number of beams (e.g., 1024).
        max_x:   Coverage limit in lateral direction (+/- meters).
        max_z:   Coverage limit in depth direction (meters).

    Returns:
        W_codebook:  (Nt x actual_size) complex matrix of weights.
        grid_coords: (actual_size x 3) matrix of focal point locations [x, y, z].
    """
    # 1. Define Grid Dimensions
    # We split the total size into (Angles x Ranges)
    # We take the square root to get an even split (e.g., 32x32 = 1024)
    N_angle = round(np.sqrt(size_cb))
    N_range = size_cb // N_angle

    # Adjust actual size to match product (in case of rounding)
    actual_size = N_angle * N_range

    # 2. Define Angular Domain (Theta)
    # Calculate max angle based on the geometry (FOV)
    # tan(theta) = x / z. We use a safe FOV of +/- 60 degrees if x=z.
    max_angle_deg = np.degrees(np.arctan(max_x / 5))
    if max_angle_deg > 75:
        max_angle_deg = 75 # Cap at 75 deg
    theta_vec = np.linspace(-max_angle_deg, max_angle_deg, N_angle)

    # 3. Define Range Domain (R)
    # We scan from very close (Fresnel region) to the max_z
    min_r = 15 # Start slightly away from array
    r_vec = np.linspace(min_r, np.sqrt(max_x**2 + max_z**2), N_range)

    # 4. Initialize Output
    W_codebook = np.zeros((Nt, actual_size), dtype=complex)
    grid_coords = np.zeros((actual_size, 3))

    col_idx = 0

    # 5. Generate Weights
    for r_val in r_vec:
        for th_val in theta_vec:
            # Convert Polar (r, theta) to Cartesian (x, z)
            # x = r * sin(theta)
            # z = r * cos(theta)
            x_foc = r_val * np.sin(np.radians(th_val))
            z_foc = r_val * np.cos(np.radians(th_val))
            y_foc = 0.0

            foc_point = np.array([[x_foc], [y_foc], [z_foc]])  # (3, 1)

            # Save coordinate
            grid_coords[col_idx, :] = foc_point.flatten()

            # --- USW Weight Calculation ---
            # 1. Distance from every antenna to this focal point
            dist_from_target_to_antennas = np.sqrt(
                np.sum((pos - foc_point) ** 2, axis=0)
            )

            # 2. Distance from array center to focal point (Reference)
            # r_center = np.linalg.norm(foc_point)

            # 3. Calculate Phase (USW Model) [Cite: Eq 20]
            # We use (d_vec - r_center) to normalize phase at the array center
            response_vector = np.exp(-1j * k * dist_from_target_to_antennas)

            # 5. Normalize Power (Norm = 1)
            w_vec = response_vector / np.linalg.norm(response_vector)

            # Store in Codebook
            W_codebook[:, col_idx] = w_vec
            col_idx += 1

    print(f"Codebook Generated: {actual_size} beams (Polar Domain).")
    return W_codebook, grid_coords
