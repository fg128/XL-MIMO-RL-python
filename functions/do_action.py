import torch

from classes.config import Config
from classes.logged_signals import LoggedSignals

import numpy as np
from numpy import ndarray


def do_action(action: ndarray, logged_signals: LoggedSignals, config: Config):
    """Uses the SAC agent to predict latents and reconstructs W and epsilon."""
    # Scale action back to VAE bounds
    latent_vector = action[:16] * 3.0           
    epsilon = (action[16] + 1.0) / 2.0      
    
    # Decode
    with torch.no_grad():
        latent_tensor = torch.tensor(latent_vector, dtype=torch.float32).unsqueeze(0)
        w_recon_scaled = config.vae.decode(latent_tensor)
        
    w_recon_flat = config.scaler.inverse_transform(w_recon_scaled.numpy())
    w_real = w_recon_flat[0, 0::2] 
    w_imag = w_recon_flat[0, 1::2] 
    W_SAC = (w_real + 1j * w_imag).reshape(config.Nt, 1)

    # Normalize power constraint
    norm_factor = np.sqrt(1 / np.sum(np.abs(W_SAC)**2))
    W = W_SAC * norm_factor
    print(W[:5])  # Print first 5 complex weights for sanity check
    print(f"Epsilon: {epsilon:.4f}")
    return W, epsilon
