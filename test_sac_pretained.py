import time
# Set path to root of project for imports
import sys

from classes.config import Config
sys.path.insert(0, '/home/student/Documents/XL-MIMO-RL-python')

import numpy as np
import torch
import joblib
from stable_baselines3 import SAC

from classes.XLMIMO_System import XLMIMO_System
from functions.visualise import visualise
from vae import XLMIMO_VAE

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
    # print(f"Observation for SAC: {obs}")
    
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

    # print(f"Action received: {action}")
    # print(f"Decoded PSF: {psf:.4f}")
    # print(f"Decoded W (first 5 values): {W[:5].flatten()}")
    return W, psf

# ==========================================
# 3. MAIN TESTING PIPELINE
# ==========================================
if __name__ == "__main__":
    # --- PATHS ---
    config = Config()
    vae_model_path = config.vae_model_path
    sac_model_path = config.sac_model_path
    scaler_path = config.scaler_path

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
        bob_random = np.array([np.random.uniform(-70, 70), 0, np.random.uniform(0, 70)])
        eve_random = bob_random + np.array([np.random.uniform(-20, 20), 0, np.random.uniform(-20, 20)])
        eve_random[0] = np.clip(eve_random[0], -70, 70)
        eve_random[2] = np.clip(eve_random[2], 0, 70)

        # Fixed locations for demonstration
        # bob_random = np.array([-52.4, 0, 44.3])
        # eve_random = np.array([32.7, 0, 13.6])

        print(f"Locations -> Bob: ({bob_random[0]:.1f}, {bob_random[2]:.1f}) | Eve: ({eve_random[0]:.1f}, {eve_random[2]:.1f})")

        xl_mimo = XLMIMO_System(bob_loc=bob_random, eve_loc=eve_random, config=config)    
        
        # ----------------------------------------------------
        # 2. Run SCA
        # ----------------------------------------------------
        print("\n[1/3] Running standard SCA Optimization...")
        t0_sca = time.perf_counter()
        W_SCA, eps_SCA, H_LUE, H_EUE, lue_coords, eue_coords = get_sca_output(xl_mimo)
        time_SCA = time.perf_counter() - t0_sca

        xl_mimo.visualize_beamfocusing(W_SCA, eps_SCA, step=1)
        sr_SCA = xl_mimo.calculate_secrecy_rate(H_LUE, H_EUE, W_SCA, eps_SCA, debug=True)
        
        SCA_stats['sr'].append(sr_SCA)
        SCA_stats['time'].append(time_SCA)
        
        print(f"--> Rate: {sr_SCA:.4f} bps/Hz | Time: {time_SCA:.4f}s")
                
        # ----------------------------------------------------
        # 3. Run VAE Compression
        # ----------------------------------------------------
        print("\n[2/3] Evaluating VAE Upper Bound...")
        t0_vae = time.perf_counter()
        W_VAE = get_vae_output(xl_mimo, W_SCA, vae, scaler)
        time_VAE = time.perf_counter() - t0_vae
        
        sr_VAE = xl_mimo.calculate_secrecy_rate(H_LUE, H_EUE, W_VAE, eps_SCA) # Uses SCA's epsilon

        visualise(W_VAE.conj(), eps_SCA, xl_mimo.bx, xl_mimo.bz, xl_mimo.ex, xl_mimo.ez, xl_mimo.config, step=2)
        
        VAE_stats['sr'].append(sr_VAE)
        VAE_stats['time'].append(time_VAE)
        
        print(f"--> Rate: {sr_VAE:.4f} bps/Hz | Time: {time_VAE:.6f}s")
        print(f"--> Degradation: {sr_SCA - sr_VAE:.4f} bps/Hz")

        # ----------------------------------------------------
        # 4. Run SAC Inference
        # ----------------------------------------------------
        print("\n[3/3] Evaluating Pre-Trained SAC Agent...")
        t0_sac = time.perf_counter()
        W_SAC, eps_SAC = get_sac_output(xl_mimo, bob_random, eve_random, vae, scaler, sac_agent=sac_agent)
        time_SAC = time.perf_counter() - t0_sac
        
        sr_SAC = xl_mimo.calculate_secrecy_rate(H_LUE, H_EUE, W_SAC, eps_SAC)

        visualise(W_SAC.conj(), eps_SAC, xl_mimo.bx, xl_mimo.bz, xl_mimo.ex, xl_mimo.ez, xl_mimo.config, step=3)
        
        SAC_stats['sr'].append(sr_SAC)
        SAC_stats['time'].append(time_SAC)
        
        print(f"--> Epsilon: {eps_SAC:.4f}")
        print(f"--> Rate: {sr_SAC:.4f} bps/Hz | Time: {time_SAC:.6f}s")
        print(f"--> Gap to VAE Ceiling: {sr_VAE - sr_SAC:.4f} bps/Hz")
        
    # Final Reporting
    print("\n========== FINAL SUMMARY ==========")
    print("--- SECRECY RATES ---")
    print(f"SCA (Ground Truth): {np.mean(SCA_stats['sr']):.4f} bps/Hz")
    print(f"VAE (Upper Bound):  {np.mean(VAE_stats['sr']):.4f} bps/Hz")
    print(f"SAC (Agent Output): {np.mean(SAC_stats['sr']):.4f} bps/Hz")
    print("\n--- INFERENCE TIMES ---")
    print(f"SCA Time: {np.mean(SCA_stats['time']):.6f} seconds")
    print(f"VAE Time: {np.mean(VAE_stats['time']):.6f} seconds")
    print(f"SAC Time: {np.mean(SAC_stats['time']):.6f} seconds")
    
    speedup = np.mean(SCA_stats['time']) / np.mean(SAC_stats['time']) if np.mean(SAC_stats['time']) > 0 else 0
    print(f"\n⚡ SAC is {speedup:.1f}x faster than SCA!")