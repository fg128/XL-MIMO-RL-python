# import numpy as np
# import matplotlib.pyplot as plt

# # ==========================================
# # 1. MOCK CONFIG & CHANNEL (For standalone running)
# # ==========================================
# class MockConfig:
#     def __init__(self):
#         self.Nt = 1024                     # Number of antennas
#         self.freq = 28e9                  # 28 GHz mmWave
#         self.c = 3e8                      # Speed of light
#         self.wavelength = self.c / self.freq
#         self.k = 2 * np.pi / self.wavelength
#         self.d = self.wavelength / 2      # Half-wavelength spacing

#         # Linear array along X-axis at Z=0
#         x_coords = np.linspace(-(self.Nt-1)*self.d/2, (self.Nt-1)*self.d/2, self.Nt)
#         self.pos = np.vstack((x_coords, np.zeros(self.Nt), np.zeros(self.Nt)))

#         self.max_x = 20
#         self.max_z = 30
#         self.resolution = 0.25             # Set to 0.1 for high-res, 0.5 for speed
#         self.noise_power_watts = 1e-13
#         self.P_total_watts = 1.0

# def mock_get_channel(config, target_pos):
#     """Calculates the Line-of-Sight Uniform Spherical Wave (USW) channel."""
#     dist = np.sqrt(np.sum((config.pos - target_pos[:, np.newaxis]) ** 2, axis=0))
#     # Simple path loss + phase shift
#     path_loss = 1 / (dist + 1e-6)
#     return path_loss * np.exp(-1j * config.k * dist)

# # ==========================================
# # 2. YOUR VISUALIZE FUNCTION (Slightly tweaked to show plot)
# # ==========================================
# def visualise_comparison(W, config, bx, bz, ex, ez, title):
#     W = np.asarray(W).reshape(-1, 1)

#     x_range = np.arange(-config.max_x, config.max_x + config.resolution, config.resolution)
#     z_range = np.arange(1, config.max_z + config.resolution, config.resolution)
#     X_grid, Z_grid = np.meshgrid(x_range, z_range)

#     probe_locs = np.vstack((X_grid.ravel(), np.zeros(X_grid.size), Z_grid.ravel()))
#     H_list = [mock_get_channel(config, probe_locs[:, i]).reshape(-1, 1) for i in range(X_grid.size)]
#     H_matrix = np.hstack(H_list)

#     rx_signals = W.conj().T @ H_matrix
#     sig_powers = np.abs(rx_signals) ** 2
#     SNR_linear = (sig_powers / config.noise_power_watts).reshape(X_grid.shape)
#     SNR_dB = 10 * np.log10(SNR_linear + 1e-30)

#     fig, ax = plt.subplots(figsize=(10, 7))
#     c = ax.pcolormesh(X_grid, Z_grid, SNR_dB, cmap='jet', shading='auto')
#     fig.colorbar(c, ax=ax, label='SNR (dB)')

#     ax.set_title(title)
#     ax.set_xlabel("Lateral (X) [m]")
#     ax.set_ylabel("Depth (Z) [m]")
#     ax.set_aspect('equal')

#     ax.plot(bx, bz, '*', markersize=15, markerfacecolor='g', markeredgecolor='k', label='BOB')
#     ax.plot(ex, ez, 'x', markersize=15, linewidth=3, color='r', label='EVE')
#     ax.plot(config.pos[0, :], config.pos[2, :], 'rs', markersize=2, label='Antennas')

#     ax.legend()
#     plt.tight_layout()
#     plt.show()

# # ==========================================
# # 3. MAIN SCRIPT: USW vs NULL-STEERING
# # ==========================================
# if __name__ == "__main__":
#     config = MockConfig()

#     # Define User Locations
#     bob_x, bob_z = 5.0, 15.0
#     eve_x, eve_z = -4.0, 10.0

#     bob_pos = np.array([bob_x, 0, bob_z])
#     eve_pos = np.array([eve_x, 0, eve_z])

#     # ---------------------------------------------------------
#     # METHOD 1: Standard USW (Just pointing at Bob)
#     # ---------------------------------------------------------
#     dist_bob = np.sqrt(np.sum((config.pos - bob_pos[:, np.newaxis]) ** 2, axis=0))
#     h_bob = np.exp(-1j * config.k * dist_bob)

#     W_standard = h_bob / np.linalg.norm(h_bob)
#     visualise_comparison(W_standard, config, bob_x, bob_z, eve_x, eve_z, "1. Standard USW (Notice Eve gets signal)")

#     # ---------------------------------------------------------
#     # METHOD 2: Virtual Null-Steering (Projecting out Eve)
#     # ---------------------------------------------------------
#     dist_eve = np.sqrt(np.sum((config.pos - eve_pos[:, np.newaxis]) ** 2, axis=0))
#     h_eve = np.exp(-1j * config.k * dist_eve)

#     # Null-Space Projection Math
#     overlap = np.vdot(h_eve, h_bob)
#     eve_norm_sq = np.vdot(h_eve, h_eve)

#     W_null_unnorm = h_bob - (overlap / eve_norm_sq) * h_eve
#     W_null = W_null_unnorm / np.linalg.norm(W_null_unnorm)

#     visualise_comparison(W_null, config, bob_x, bob_z, eve_x, eve_z, "2. Null-Steered USW (Notice the blue deadzone on Eve)")

import numpy as np

def calculate_analytical_secrecy_rate(W, h_bob, h_eve, V, P_s, P_an, noise_power):
    """
    Uses your exact step_function logic to calculate the Secrecy Rate
    for a given beamforming weight matrix W.
    """
    # 1. Analytical signal power
    # Using np.vdot is mathematically identical to (h.conj().T @ W).item() but safer for arrays
    sig_pwr_bob = P_s * np.abs(np.vdot(h_bob, W))**2
    sig_pwr_eve = P_s * np.abs(np.vdot(h_eve, W))**2

    # 2. Analytical interference power (AN leakage)
    # V is the null-space of Bob, so AN leaks heavily to Eve but 0 to Bob
    an_leakage_bob = P_an * (np.linalg.norm(h_bob.conj().T @ V) ** 2)
    an_leakage_eve = P_an * (np.linalg.norm(h_eve.conj().T @ V) ** 2)

    # 3. SINR
    SINR_bob = sig_pwr_bob / (noise_power + an_leakage_bob)
    SINR_eve = sig_pwr_eve / (noise_power + an_leakage_eve)

    # 4. Secrecy rate
    rate_bob = np.log2(1 + SINR_bob)
    rate_eve = np.log2(1 + SINR_eve)
    secrecy_rate = max(0.0, rate_bob - rate_eve)

    return secrecy_rate, rate_bob, rate_eve

# ==========================================
# TEST COMPARISON SCRIPT
# ==========================================
if __name__ == "__main__":
    # --- Mock Setup (Replace with your actual config/locations) ---
    Nt = 1024
    k = 2 * np.pi / (3e8 / 28e9)
    pos = np.vstack((np.linspace(-1, 1, Nt), np.zeros(Nt), np.zeros(Nt)))
    noise_power_watts = 1e-13
    P_total_watts = 0.1
    psf = 1 # 80% power to signal, 20% to AN

    P_s = P_total_watts * psf
    P_an = P_total_watts * (1 - psf)

    # Mock Locations
    bob_loc = np.array([5.0, 0, 15.0])
    eve_loc = np.array([4.0, 0, 12.0])

    # --- 1. Get Channels ---
    dist_bob = np.sqrt(np.sum((pos - bob_loc[:, np.newaxis]) ** 2, axis=0))
    dist_eve = np.sqrt(np.sum((pos - eve_loc[:, np.newaxis]) ** 2, axis=0)) + np.random.rand() * 10  # Add small random to avoid perfect alignment

    # Shape to (Nt, 1) to match your environment
    h_bob = (np.exp(-1j * k * dist_bob) / dist_bob).reshape(-1, 1)
    h_eve = (np.exp(-1j * k * dist_eve) / dist_eve).reshape(-1, 1)

    # --- 2. Calculate AN Matrix V (Same for both setups) ---
    # Null space of Bob ensures AN only hurts Eve
    V = np.eye(Nt) - (h_bob @ h_bob.conj().T) / (np.linalg.norm(h_bob)**2)

    # ==========================================
    # SETUP 1: Standard USW
    # ==========================================
    # Focus directly on Bob
    W_std = h_bob / np.linalg.norm(h_bob)

    sr_std, r_bob_std, r_eve_std = calculate_analytical_secrecy_rate(
        W_std, h_bob, h_eve, V, P_s, P_an, noise_power_watts
    )

    # ==========================================
    # SETUP 2: Virtual Null-Steered USW
    # ==========================================
    # Project Eve out of Bob's beam
    overlap = np.vdot(h_eve, h_bob)
    eve_norm_sq = np.vdot(h_eve, h_eve)

    W_null_unnorm = h_bob - (overlap / eve_norm_sq) * h_eve
    W_null = W_null_unnorm / np.linalg.norm(W_null_unnorm)

    sr_null, r_bob_null, r_eve_null = calculate_analytical_secrecy_rate(
        W_null, h_bob, h_eve, V, P_s, P_an, noise_power_watts
    )

    # --- Results ---
    print(f"--- STANDARD USW ---")
    print(f"Bob Rate:     {r_bob_std:.4f} bps/Hz")
    print(f"Eve Rate:     {r_eve_std:.4f} bps/Hz")
    print(f"SECRECY RATE: {sr_std:.4f} bps/Hz\n")

    print(f"--- NULL-STEERED USW ---")
    print(f"Bob Rate:     {r_bob_null:.4f} bps/Hz")
    print(f"Eve Rate:     {r_eve_null:.4f} bps/Hz")
    print(f"SECRECY RATE: {sr_null:.4f} bps/Hz")
