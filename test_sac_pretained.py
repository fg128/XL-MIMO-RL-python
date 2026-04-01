import time
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
from stable_baselines3 import SAC

from functions.get_channel import get_channel
from functions.visualise import visualise
from classes.config import Config
from vae import XLMIMO_VAE

# ==========================================
# 1. THE XL-MIMO SYSTEM 
# ==========================================
class XLMIMO_System:
    def __init__(self, bob_loc, eve_loc, N_bx=32, N_bz=1, K=1, E=1, freq=2e9, P_b_dBm=20, noise_dBm=-96):
        self.N_bx = N_bx
        self.N_bz = N_bz
        self.N_b = N_bx * N_bz
        self.K = K
        self.E = E
        self.bob_r = np.sqrt(bob_loc[0]**2 + bob_loc[2]**2)
        self.bob_theta = np.arctan2(bob_loc[2], bob_loc[0])
        self.eve_r = np.sqrt(eve_loc[0]**2 + eve_loc[2]**2)
        self.eve_theta = np.arctan2(eve_loc[2], eve_loc[0])

        self.c = 299_792_458 
        self.freq = freq
        self.lambda_c = self.c / self.freq
        self.d_ant = self.lambda_c / 2

        self.P_b = 10 ** ((P_b_dBm - 30) / 10)
        self.sigma2 = 10 ** ((noise_dBm - 30) / 10)

        self.config = Config()
        self.bx = bob_loc[0]
        self.bz = bob_loc[2]
        self.ex = eve_loc[0]
        self.ez = eve_loc[2]

    def generate_channels(self):
        H_LUE = np.zeros((self.K, self.N_b), dtype=complex)
        H_EUE = np.zeros((self.E, self.N_b), dtype=complex)
        lue_coords, eue_coords = [], []

        # 3. Generate channels for Bob
        for k in range(self.K):
            # Formulate Bob's 3D coordinates based on initialization
            bob_loc = [self.bx, 0, self.bz] 
            H_LUE[k, :] = get_channel(self.config, bob_loc).flatten()
            lue_coords.append((self.bob_theta, self.bob_r))

        # 4. Generate channels for Eve
        for e in range(self.E):
            # Formulate Eve's 3D coordinates based on initialization
            eve_loc = [self.ex, 0, self.ez]
            H_EUE[e, :] = get_channel(self.config, eve_loc).flatten()
            eue_coords.append((self.eve_theta, self.eve_r))

        return H_LUE, H_EUE, lue_coords, eue_coords

    def calculate_secrecy_rate(self, H_LUE, H_EUE, W, epsilon, debug=False):
        min_rate = np.inf
        H_b_pinv = np.linalg.pinv(H_LUE)
        V_0 = np.eye(self.N_b) - H_b_pinv @ H_LUE
        AN_power = (1 - epsilon) * self.P_b

        for k in range(self.K):
            for e in range(self.E):
                signal_k = epsilon * self.P_b * np.abs(H_LUE[k, :] @ W[:, k])**2
                interf_k = epsilon * self.P_b * sum(np.abs(H_LUE[k, :] @ W[:, j])**2 for j in range(self.K) if j != k)
                if debug:
                    print(f"Signal Power at Bob: {signal_k:.4e} W | IAN Leakage at Bob: {interf_k:.4e} W | Noise Power: {self.sigma2:.4e} W")
                R_k = np.log2(1 + signal_k / (interf_k + self.sigma2))

                signal_e = epsilon * self.P_b * np.abs(H_EUE[e, :] @ W[:, k])**2
                interf_e = epsilon * self.P_b * sum(np.abs(H_EUE[e, :] @ W[:, j])**2 for j in range(self.K) if j != k)
                noise_an_e = AN_power * np.linalg.norm(H_EUE[e, :] @ V_0)**2
                R_e = np.log2(1 + signal_e / (interf_e + noise_an_e + self.sigma2))

                if debug:
                    print(f"Bob Rate: {R_k:.4f} bps/Hz | Eve Rate: {R_e:.4f} bps/Hz")

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
        """Now strictly returns W and epsilon without calculating internal secrecy rate."""
        H_LUE, H_EUE, lue_coords, eue_coords = self.generate_channels()
        H_b_conj = H_LUE.conj().T
        W_t = H_b_conj @ np.linalg.inv(H_LUE @ H_b_conj + self.sigma2 * np.eye(self.K))
        W_t = W_t * np.sqrt(self.K / np.trace(W_t.conj().T @ W_t).real)

        epsilon_t = 0.5
        for t in range(1, 15):
            W_next = self.optimize_beamfocusing_SCA(H_LUE, H_EUE, W_t, epsilon_t)
            if W_next is not None: W_t = W_next
            epsilon_t = self.optimize_power_allocation_GSS(H_LUE, H_EUE, W_t)

        return W_t, epsilon_t, H_LUE, H_EUE, lue_coords, eue_coords

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
        
        filename = title.replace(" ", "_").lower() + ".png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close() 

# ==========================================
# 2. PURE GENERATION FUNCTIONS
# ==========================================
def get_sca_output(xl_mimo):
    """Runs the SCA optimization baseline and returns only the core parameters."""
    W, epsilon, H_LUE, H_EUE, lue_coords, eue_coords = xl_mimo.run_alternating_optimization()
    return W, epsilon, H_LUE, H_EUE, lue_coords, eue_coords

def get_vae_output(xl_mimo, W_SCA, vae, scaler):
    """Compresses the SCA output through the VAE and reconstructs W."""
    w_flat = np.column_stack((W_SCA.real.flatten(), W_SCA.imag.flatten())).flatten().reshape(1, -1)
    w_scaled = scaler.transform(w_flat)
    tensor_in = torch.tensor(w_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        w_recon_scaled, _, _ = vae(tensor_in)
        
    w_recon_flat = scaler.inverse_transform(w_recon_scaled.numpy())
    w_real_recon = w_recon_flat[0, 0::2] 
    w_imag_recon = w_recon_flat[0, 1::2] 
    W_VAE = (w_real_recon + 1j * w_imag_recon).reshape(xl_mimo.N_b, xl_mimo.K)

    # Normalize power constraint
    norm_factor = np.sqrt(xl_mimo.K / np.sum(np.abs(W_VAE)**2))
    W = W_VAE * norm_factor
    return W

def get_sac_output(xl_mimo, bob_loc, eve_loc, vae, scaler, sac_agent=None, action=None):
    """Uses the SAC agent to predict latents and reconstructs W and epsilon."""
    # Format state/observation (Scale by 70.0 assuming training environment bounds)
    obs = np.array([bob_loc[0]/70.0, bob_loc[2]/70.0, eve_loc[0]/70.0, eve_loc[2]/70.0], dtype=np.float32)
    print(f"Observation for SAC: {obs}")
    
    # Predict action [-1, 1]
    if action is None:
        action, _ = sac_agent.predict(obs, deterministic=True)
    
    # Scale action back to VAE bounds
    latent_vector = action[:16] * 3.0           
    psf = (action[16] + 1.0) / 2.0      
    
    # Decode
    with torch.no_grad():
        latent_tensor = torch.tensor(latent_vector, dtype=torch.float32).unsqueeze(0)
        w_recon_scaled = vae.decode(latent_tensor)
        
    w_recon_flat = scaler.inverse_transform(w_recon_scaled.numpy())
    w_real = w_recon_flat[0, 0::2] 
    w_imag = w_recon_flat[0, 1::2] 
    W_SAC = (w_real + 1j * w_imag).reshape(256, 1)

    # Normalize power constraint
    norm_factor = np.sqrt(1 / np.sum(np.abs(W_SAC)**2))
    W = W_SAC * norm_factor

    print(f"Action received: {action}")
    print(f"Decoded PSF: {psf:.4f}")
    print(f"Decoded W (first 5 values): {W[:5].flatten()}")
    return W, psf

# ==========================================
# 3. MAIN TESTING PIPELINE
# ==========================================
if __name__ == "__main__":
    # --- PATHS ---
    vae_model_path = 'checkpoints/w_matrix_full_vae_epoch_400.pth'
    sac_model_path = 'sac_pretrained_brain.zip'
    scaler_path = 'xlmimo_w_scaler.pkl'

    # --- LOAD MODELS ---
    print("\n[0/3] Loading VAE, Scaler, and Pre-Trained SAC Agent...")
    scaler = joblib.load(scaler_path)
    
    vae = XLMIMO_VAE(input_dim=512, latent_dim=16)
    vae.load_state_dict(torch.load(vae_model_path, map_location=torch.device('cpu')))
    vae.eval()

    sac_agent = SAC.load(sac_model_path, device='cpu')
    print("[OK] Models loaded successfully.")

    # Tracking lists
    SCA_stats = {'sr': [], 'time': []}
    VAE_stats = {'sr': [], 'time': []}
    SAC_stats = {'sr': [], 'time': []}

    for trial in range(1):
        print(f"\n========== TRIAL {trial+1} ==========")
        
        # 1. Setup Environment
        # Pure Random Locations within bounds for Bob and Eve
        # bob_random = np.array([np.random.uniform(-70, 70), 0, np.random.uniform(0, 70)])
        # eve_random = np.array([np.random.uniform(-70, 70), 0, np.random.uniform(0, 70)])

        # Random but with a minimum distance constraint (e.g., 20m apart)
        # bob_random = np.array([np.random.uniform(-70, 70), 0, np.random.uniform(0, 70)])
        # eve_random = bob_random + np.array([np.random.uniform(-20, 20), 0, np.random.uniform(-20, 20)])
        # eve_random[0] = np.clip(eve_random[0], -70, 70)
        # eve_random[2] = np.clip(eve_random[2], 0, 70)

        # Fixed locations for demonstration
        bob_random = np.array([-52.4, 0, 44.3])
        eve_random = np.array([32.7, 0, 13.6])

        print(f"Locations -> Bob: ({bob_random[0]:.1f}, {bob_random[2]:.1f}) | Eve: ({eve_random[0]:.1f}, {eve_random[2]:.1f})")

        xl_mimo = XLMIMO_System(N_bx=256, N_bz=1, K=1, E=1, bob_loc=bob_random, eve_loc=eve_random)    
        
        # ----------------------------------------------------
        # 2. Run SCA
        # ----------------------------------------------------
        print("\n[1/3] Running standard SCA Optimization...")
        t0_sca = time.perf_counter()
        W_SCA, eps_SCA, H_LUE, H_EUE, lue_coords, eue_coords = get_sca_output(xl_mimo)
        time_SCA = time.perf_counter() - t0_sca

        np.save('my_data.npy', W_SCA)

        visualise(W_SCA.conj(), eps_SCA, xl_mimo.bx, xl_mimo.bz, xl_mimo.ex, xl_mimo.ez, xl_mimo.config, step=1)
        sr_SCA = xl_mimo.calculate_secrecy_rate(H_LUE, H_EUE, W_SCA, eps_SCA, debug=True)
        
        SCA_stats['sr'].append(sr_SCA)
        SCA_stats['time'].append(time_SCA)
        
        print(f"--> Rate: {sr_SCA:.4f} bps/Hz | Time: {time_SCA:.4f}s")
                
    #     # ----------------------------------------------------
    #     # 3. Run VAE Compression
    #     # ----------------------------------------------------
    #     print("\n[2/3] Evaluating VAE Upper Bound...")
    #     t0_vae = time.perf_counter()
    #     W_VAE = get_vae_output(xl_mimo, W_SCA, vae, scaler)
    #     time_VAE = time.perf_counter() - t0_vae
        
    #     sr_VAE = xl_mimo.calculate_secrecy_rate(H_LUE, H_EUE, W_VAE, eps_SCA) # Uses SCA's epsilon

    #     visualise(W_VAE.conj(), eps_SCA, xl_mimo.bx, xl_mimo.bz, xl_mimo.ex, xl_mimo.ez, xl_mimo.config, step=2)
        
    #     VAE_stats['sr'].append(sr_VAE)
    #     VAE_stats['time'].append(time_VAE)
        
    #     print(f"--> Rate: {sr_VAE:.4f} bps/Hz | Time: {time_VAE:.6f}s")
    #     print(f"--> Degradation: {sr_SCA - sr_VAE:.4f} bps/Hz")

    #     # ----------------------------------------------------
    #     # 4. Run SAC Inference
    #     # ----------------------------------------------------
    #     print("\n[3/3] Evaluating Pre-Trained SAC Agent...")
    #     t0_sac = time.perf_counter()
    #     W_SAC, eps_SAC = get_sac_output(xl_mimo, bob_random, eve_random, vae, scaler, sac_agent=sac_agent)
    #     time_SAC = time.perf_counter() - t0_sac
        
    #     sr_SAC = xl_mimo.calculate_secrecy_rate(H_LUE, H_EUE, W_SAC, eps_SAC)

    #     visualise(W_SAC.conj(), eps_SAC, xl_mimo.bx, xl_mimo.bz, xl_mimo.ex, xl_mimo.ez, xl_mimo.config, step=3)
        
    #     SAC_stats['sr'].append(sr_SAC)
    #     SAC_stats['time'].append(time_SAC)
        
    #     print(f"--> Epsilon: {eps_SAC:.4f}")
    #     print(f"--> Rate: {sr_SAC:.4f} bps/Hz | Time: {time_SAC:.6f}s")
    #     print(f"--> Gap to VAE Ceiling: {sr_VAE - sr_SAC:.4f} bps/Hz")
        
    # # Final Reporting
    # print("\n========== FINAL SUMMARY ==========")
    # print("--- SECRECY RATES ---")
    # print(f"SCA (Ground Truth): {np.mean(SCA_stats['sr']):.4f} bps/Hz")
    # print(f"VAE (Upper Bound):  {np.mean(VAE_stats['sr']):.4f} bps/Hz")
    # print(f"SAC (Agent Output): {np.mean(SAC_stats['sr']):.4f} bps/Hz")
    # print("\n--- INFERENCE TIMES ---")
    # print(f"SCA Time: {np.mean(SCA_stats['time']):.6f} seconds")
    # print(f"VAE Time: {np.mean(VAE_stats['time']):.6f} seconds")
    # print(f"SAC Time: {np.mean(SAC_stats['time']):.6f} seconds")
    
    # speedup = np.mean(SCA_stats['time']) / np.mean(SAC_stats['time']) if np.mean(SAC_stats['time']) > 0 else 0
    # print(f"\n⚡ SAC is {speedup:.1f}x faster than SCA!")