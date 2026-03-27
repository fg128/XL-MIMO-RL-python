import time
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib

# ==========================================
# 1. THE VAE ARCHITECTURE (Needed for loading)
# ==========================================
class XLMIMO_VAE(nn.Module):
    def __init__(self, input_dim=512, latent_dim=16):
        super(XLMIMO_VAE, self).__init__()
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        h = self.encoder_base(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ==========================================
# 2. THE XL-MIMO SYSTEM
# ==========================================
class XLMIMO_System:
    def __init__(self, bob_loc, eve_loc, N_bx=32, N_bz=1, K=1, E=1, freq=2e9, P_b_dBm=20, noise_dBm=-96):
        self.N_bx = N_bx
        self.N_bz = N_bz
        self.N_b = N_bx * N_bz
        self.K = K
        self.E = E
        self.bob_r = np.sqrt(bob_loc[0]**2 + bob_loc[2]**2)
        self.bob_theta = np.arctan2(bob_loc[0], bob_loc[2])
        self.eve_r = np.sqrt(eve_loc[0]**2 + eve_loc[2]**2)
        self.eve_theta = np.arctan2(eve_loc[0], eve_loc[2])

        self.c = 299_792_458 
        self.freq = freq
        self.lambda_c = self.c / self.freq
        self.d_ant = self.lambda_c / 2

        self.P_b = 10 ** ((P_b_dBm - 30) / 10)
        self.sigma2 = 10 ** ((noise_dBm - 30) / 10)

    def near_field_array_response(self, theta, phi, r):
        a = np.zeros(self.N_b, dtype=complex)
        idx = 0
        for nx in range(-(self.N_bx//2), self.N_bx//2 + 1 if self.N_bx%2!=0 else self.N_bx//2):
            for nz in range(-(self.N_bz//2), self.N_bz//2 + 1 if self.N_bz%2!=0 else self.N_bz//2):
                dist_shift = (nx * self.d_ant * np.cos(theta) * np.sin(phi) +
                              nz * self.d_ant * np.cos(phi))
                phase = - (2 * np.pi / self.lambda_c) * (r - dist_shift)
                a[idx] = np.exp(1j * phase)
                idx += 1
        return a

    def generate_channels(self):
        H_LUE = np.zeros((self.K, self.N_b), dtype=complex)
        H_EUE = np.zeros((self.E, self.N_b), dtype=complex)
        lue_coords, eue_coords = [], []
        phi = np.pi / 2

        for k in range(self.K):
            theta, r = self.bob_theta, self.bob_r
            H_LUE[k, :] = self.near_field_array_response(theta, phi, r) * np.sqrt(1e-7)
            lue_coords.append((theta, r))

        for e in range(self.E):
            theta, r = self.eve_theta, self.eve_r
            H_EUE[e, :] = self.near_field_array_response(theta, phi, r) * np.sqrt(1e-7)
            eue_coords.append((theta, r))

        return H_LUE, H_EUE, lue_coords, eue_coords

    def calculate_secrecy_rate(self, H_LUE, H_EUE, W, epsilon):
        min_rate = np.inf
        H_b_pinv = np.linalg.pinv(H_LUE)
        V_0 = np.eye(self.N_b) - H_b_pinv @ H_LUE
        AN_power = (1 - epsilon) * self.P_b / self.N_b

        for k in range(self.K):
            for e in range(self.E):
                signal_k = epsilon * self.P_b * np.abs(H_LUE[k, :] @ W[:, k])**2
                interf_k = epsilon * self.P_b * sum(np.abs(H_LUE[k, :] @ W[:, j])**2 for j in range(self.K) if j != k)
                R_k = np.log2(1 + signal_k / (interf_k + self.sigma2))

                signal_e = epsilon * self.P_b * np.abs(H_EUE[e, :] @ W[:, k])**2
                interf_e = epsilon * self.P_b * sum(np.abs(H_EUE[e, :] @ W[:, j])**2 for j in range(self.K) if j != k)
                noise_an_e = AN_power * np.linalg.norm(H_EUE[e, :] @ V_0)**2
                R_e = np.log2(1 + signal_e / (interf_e + noise_an_e + self.sigma2))

                secrecy_rate = max(0, R_k - R_e)
                if secrecy_rate < min_rate:
                    min_rate = secrecy_rate
        return min_rate

    def optimize_power_allocation_GSS(self, H_LUE, H_EUE, W):
        phi_gss = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi_gss
        a, b = 0.001, 0.999
        tol = 0.05 * (b - a)
        c, d = a + resphi * (b - a), b - resphi * (b - a)
        fc = self.calculate_secrecy_rate(H_LUE, H_EUE, W, c)
        fd = self.calculate_secrecy_rate(H_LUE, H_EUE, W, d)

        while abs(b - a) > tol:
            if fc > fd:
                b, d, fd = d, c, fc
                c = a + resphi * (b - a)
                fc = self.calculate_secrecy_rate(H_LUE, H_EUE, W, c)
            else:
                a, c, fc = c, d, fd
                d = b - resphi * (b - a)
                fd = self.calculate_secrecy_rate(H_LUE, H_EUE, W, d)
        return (b + a) / 2

    def optimize_beamfocusing_SCA(self, H_LUE, H_EUE, W_prev, epsilon):
        W = cp.Variable((self.N_b, self.K), complex=True)
        xi = cp.Variable(nonneg=True)
        constraints = [cp.sum(cp.sum_squares(cp.abs(W))) <= self.K]

        P_W = epsilon * self.P_b
        P_AN = (1 - epsilon) * self.P_b / self.N_b
        H_b_pinv = np.linalg.pinv(H_LUE)
        V_0 = np.eye(self.N_b) - H_b_pinv @ H_LUE

        for k in range(self.K):
            h_b_k = np.sqrt(P_W) * H_LUE[k, :]
            sigma_LUE = self.sigma2 

            h_b_w_t_k = h_b_k @ W_prev[:, k]
            signal_b_t = np.abs(h_b_w_t_k)**2
            interf_b_t = sum(np.abs(h_b_k @ W_prev[:, j])**2 for j in range(self.K) if j != k)
            total_b_t = signal_b_t + interf_b_t
            R_kk_t = np.log(1 + signal_b_t / (interf_b_t + sigma_LUE))

            T1_b = R_kk_t
            T2_b = - (signal_b_t + sigma_LUE) / (interf_b_t + sigma_LUE)
            T3_b = sigma_LUE / (total_b_t + sigma_LUE)
            T4_b = (2 * cp.real(np.conj(h_b_w_t_k) * (h_b_k @ W[:, k]))) / (interf_b_t + sigma_LUE)
            current_total_b_W = cp.sum([cp.sum_squares(h_b_k @ W[:, j]) for j in range(self.K)])
            T5_b = - (signal_b_t * current_total_b_W) / ((total_b_t + sigma_LUE) * (interf_b_t + sigma_LUE))
            f1_LB = T1_b + T2_b + T3_b + T4_b + T5_b

            for e in range(self.E):
                h_e_e = np.sqrt(P_W) * H_EUE[e, :]
                sigma_EUE = P_AN * np.linalg.norm(H_EUE[e, :] @ V_0)**2 + self.sigma2

                h_e_w_t_k = h_e_e @ W_prev[:, k]
                signal_e_t = np.abs(h_e_w_t_k)**2
                interf_e_t = sum(np.abs(h_e_e @ W_prev[:, j])**2 for j in range(self.K) if j != k)
                total_e_t = signal_e_t + interf_e_t
                R_ek_t = np.log(1 + signal_e_t / (interf_e_t + sigma_EUE))

                T1_e = R_ek_t
                T2_e = interf_e_t / sigma_EUE
                T3_e = - (signal_e_t * sigma_EUE) / ((total_e_t + sigma_EUE) * (interf_e_t + sigma_EUE))
                current_total_e_W = cp.sum([cp.sum_squares(h_e_e @ W[:, j]) for j in range(self.K)])
                T4_e = current_total_e_W / (total_e_t + sigma_EUE)
                trace_expansion = cp.sum([np.conj(h_e_e @ W_prev[:, j]) * (h_e_e @ W[:, j]) for j in range(self.K) if j != k])
                T5_e = - (2 * cp.real(trace_expansion)) / sigma_EUE
                current_interf_e_W = cp.sum([cp.sum_squares(h_e_e @ W[:, j]) for j in range(self.K) if j != k])
                T6_e = (interf_e_t * current_interf_e_W) / ((interf_e_t + sigma_EUE) * sigma_EUE)
                f2_UB = T1_e + T2_e + T3_e + T4_e + T5_e + T6_e

                LB_R_k = f1_LB / np.log(2)
                UB_R_ek = f2_UB / np.log(2)
                constraints.append(LB_R_k - UB_R_ek >= xi)

        prob = cp.Problem(cp.Maximize(xi), constraints)
        try:
            prob.solve(solver=cp.CLARABEL, max_iters=2000)
            return W.value
        except Exception as e:
            return W_prev

    def run_alternating_optimization(self):
        H_LUE, H_EUE, lue_coords, eue_coords = self.generate_channels()
        H_b_conj = H_LUE.conj().T
        W_t = H_b_conj @ np.linalg.inv(H_LUE @ H_b_conj + self.sigma2 * np.eye(self.K))
        W_t = W_t * np.sqrt(self.K / np.trace(W_t.conj().T @ W_t).real)

        epsilon_t = 0.5
        for t in range(1, 15):
            W_next = self.optimize_beamfocusing_SCA(H_LUE, H_EUE, W_t, epsilon_t)
            if W_next is not None: W_t = W_next
            epsilon_t = self.optimize_power_allocation_GSS(H_LUE, H_EUE, W_t)
            current_rate = self.calculate_secrecy_rate(H_LUE, H_EUE, W_t, epsilon_t)

        # RETURN THE CHANNELS AND COORDS SO WE CAN REUSE THEM FOR THE VAE TEST
        return W_t, epsilon_t, current_rate, H_LUE, H_EUE, lue_coords, eue_coords

    def visualize_beamfocusing(self, W, epsilon, lue_coords, eue_coords, title="Near-Field Optimal Beamfocusing Heatmap"):
        print(f"Generating visualization: {title}... ")
        x_min, x_max, y_min, y_max, grid_res = -70, 70, 2, 20, 50
        X, Y = np.meshgrid(np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res))
        Power_Map = np.zeros_like(X, dtype=float)
        P_W = epsilon * self.P_b

        for i in range(grid_res):
            for j in range(grid_res):
                x, y = X[i, j], Y[i, j]
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                phi = np.pi / 2 
                h_test = self.near_field_array_response(theta, phi, r) * np.sqrt(1e-7)
                p_signal = sum(P_W * np.abs(h_test @ W[:, k])**2 for k in range(self.K))
                Power_Map[i, j] = p_signal

        plt.figure(figsize=(10, 7))
        plt.pcolormesh(X, Y, 10*np.log10(Power_Map + 1e-12), shading='auto', cmap='viridis')
        plt.colorbar(label='Signal Power (dB)')

        for idx, (theta, r) in enumerate(lue_coords):
            plt.scatter(r * np.cos(theta), r * np.sin(theta), c='lime', marker='^', s=150, edgecolors='black', label='Bob (LUE)' if idx == 0 else "")
        for idx, (theta, r) in enumerate(eue_coords):
            plt.scatter(r * np.cos(theta), r * np.sin(theta), c='red', marker='X', s=150, edgecolors='black', label='Eve (EUE)' if idx == 0 else "")

        plt.scatter(0, 0, c='cyan', marker='s', s=200, edgecolors='black', label='Base Station')
        plt.title(f"{title}\nEpsilon = {epsilon:.3f}")
        plt.xlabel("X coordinate (m)")
        plt.ylabel("Y/Depth coordinate (m)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # With this:
        filename = title.replace(" ", "_").lower() + ".png"
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved plot as {filename}")
        plt.close() # Close the figure to free up memory


# ==========================================
# 3. MAIN TESTING PIPELINE
# ==========================================
if __name__ == "__main__":
    # --- 1. RUN THE BASELINE (SCA) ---
    print("\n[1/3] Running standard SCA Optimization...")
    # Generate random X (-15 to 15) and Z (2 to 20), keeping Y = 0
    bob_random = np.array([np.random.uniform(2, 20), 0, np.random.uniform(-20, 20)])
    eve_random = np.array([np.random.uniform(2, 20), 0, np.random.uniform(-20, 20)])
    print(f"Random Bob Location: ({bob_random[0]:.1f}, {bob_random[2]:.1f}) | Random Eve Location: ({eve_random[0]:.1f}, {eve_random[2]:.1f})")

    # Initialize the system with the new random locations
    xl_mimo = XLMIMO_System(N_bx=256, N_bz=1, K=1, E=1, bob_loc=bob_random, eve_loc=eve_random)    
    W_SCA, epsilon_SCA, sr_SCA, H_LUE, H_EUE, lue_coords, eue_coords = xl_mimo.run_alternating_optimization()
    
    print(f"--> SCA Final Secrecy Rate: {sr_SCA:.4f} bps/Hz")
    xl_mimo.visualize_beamfocusing(W_SCA, epsilon_SCA, lue_coords, eue_coords, title="ORIGINAL SCA SOLVER")

    # --- 2. LOAD VAE & SCALER ---
    print("\n[2/3] Loading VAE and compressing the SCA beam...")
    scaler = joblib.load('xlmimo_w_scaler.pkl')
    vae = XLMIMO_VAE(input_dim=512, latent_dim=16)
    vae.load_state_dict(torch.load('w_matrix_full_vae.pth', map_location=torch.device('cpu')))
    vae.eval()

    # *CRITICAL NOTE FIXED: Interleaved Format (Real0, Imag0, Real1, Imag1...)*
    
    # 1. Flattening the SCA output to match your CSV
    # We take the Reals and Imags, stack them side-by-side, and flatten them into a 1D array
    w_flat = np.column_stack((W_SCA.real.flatten(), W_SCA.imag.flatten())).flatten().reshape(1, -1)
    
    # Scale -> Tensor -> VAE -> Unscale
    w_scaled = scaler.transform(w_flat)
    tensor_in = torch.tensor(w_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        w_recon_scaled, mu, logvar = vae(tensor_in)
        
    w_recon_flat = scaler.inverse_transform(w_recon_scaled.numpy())

    # 2. Rebuilding the Complex Matrix from Interleaved Data
    # Slice the array: Start at 0, take every 2nd step (Evens = Reals)
    w_real_recon = w_recon_flat[0, 0::2] 
    # Slice the array: Start at 1, take every 2nd step (Odds = Imags)
    w_imag_recon = w_recon_flat[0, 1::2] 
    
    W_VAE = (w_real_recon + 1j * w_imag_recon).reshape(256, 1)

    # --- 3. EVALUATE THE VAE RECONSTRUCTION ---
    print("\n[3/3] Evaluating VAE Reconstructed Beam...")
    
    # Calculate the secrecy rate using the reconstructed beam
    sr_VAE = xl_mimo.calculate_secrecy_rate(H_LUE, H_EUE, W_VAE, epsilon_SCA)
    
    print(f"--> VAE Reconstructed Secrecy Rate: {sr_VAE:.4f} bps/Hz")
    print(f"--> Rate Degradation: {sr_SCA - sr_VAE:.4f} bps/Hz")
    
    # Generate the side-by-side heatmap
    xl_mimo.visualize_beamfocusing(W_VAE, epsilon_SCA, lue_coords, eue_coords, title="VAE RECONSTRUCTED BEAM")

    plt.show() # Keeps all plots open at the end