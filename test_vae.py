import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from functions.visualise import visualise
from vae import XLMIMO_VAE


def run_sanity_check():
    print("--- Loading Artifacts ---")
    
    # Load the Scaler
    try:
        scaler = joblib.load('xlmimo_w_scaler.pkl')
        print("[OK] Scaler loaded.")
    except Exception as e:
        print(f"[FAIL] Could not load scaler: {e}")
        return

    # Load the Model
    model = XLMIMO_VAE(input_dim=512, latent_dim=16)
    try:
        model.load_state_dict(torch.load('checkpoints/w_matrix_full_vae_epoch_1000.pth', map_location=torch.device('cpu')))
        model.eval() # Set to evaluation mode (turns off random dropout/batchnorm if you had them)
        print("[OK] VAE Model weights loaded.")
    except Exception as e:
        print(f"[FAIL] Could not load model weights: {e}")
        return

    print("\n--- TEST 1: The Reconstruction Test ---")
    # Grab just the first row of your CSV to test
    df = pd.read_csv("xlmimo_oracle_dataset.csv", nrows=1)
    
    cols_to_drop = ['Episode', 'Bob_X', 'Bob_Z', 'Eve_X', 'Eve_Z', 'Epsilon', 'Secrecy_Rate']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df_w_matrix = df.drop(columns=existing_cols_to_drop)
    
    original_w_raw = df_w_matrix.values.astype(np.float32)
    
    # Scale it, encode it, decode it
    original_w_scaled = scaler.transform(original_w_raw)
    tensor_in = torch.tensor(original_w_scaled)
    
    with torch.no_grad():
        reconstructed_w_scaled, mu, logvar = model(tensor_in)
                
    # Un-scale it back to physical reality
    reconstructed_w_raw = scaler.inverse_transform(reconstructed_w_scaled.numpy())
    
    print("\nComparing the first 5 complex weight pairs (Real, Imaginary):")
    print(f"{'Index':<10} | {'Original (SCA)':<25} | {'Reconstructed (VAE)':<25}")
    print("-" * 65)
    for i in range(10): # First 10 values (5 complex numbers)
        orig_val = original_w_raw[0][i]
        recon_val = reconstructed_w_raw[0][i]
        print(f"Weight [{i}] | {orig_val:>20.6f} | {recon_val:>20.6f}")


    print("\n--- TEST 2: The RL Latent Simulation ---")
    # Simulate what the SAC Actor will output: A random 16D vector bounded mostly between -3 and 3
    print("Simulating a random 16-dimensional action from the SAC agent...")
    simulated_action = torch.randn(1, 16) * 2.0 # Standard normal distribution scaled slightly
    
    with torch.no_grad():
        generated_w_scaled = model.decode(simulated_action)
        
    generated_w_raw = scaler.inverse_transform(generated_w_scaled.numpy())
    
    print("\nGenerated Physical Weights (First 5 pairs):")
    for i in range(10):
        print(f"Gen Weight [{i}]: {generated_w_raw[0][i]:.6f}")
        
    print("\n[SUCCESS] The Decoder is ready to be plugged into the environment!")

if __name__ == "__main__":
    run_sanity_check()