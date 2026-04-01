import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import SAC
from torch.utils.data import TensorDataset, DataLoader

# Import your environment and config
from classes.config import Config
from XL_MIMO_Enviroment import XLMIMOEnv
from vae import XLMIMO_VAE
# Import your VAE model (adjust the import path based on your project structure)
# from classes.vae_model import VAE 

# -------------------------------------------------------------------------
# 1. LOAD THE ORACLE DATASET
# -------------------------------------------------------------------------
print("Loading Oracle Dataset...")
df = pd.read_csv("xlmimo_oracle_dataset.csv")

# Extract the State (Observation)
# Adjust these columns if your env's observation space is different!
# Assumes observation is: [Bob_X, Bob_Z, Eve_X, Eve_Z]
X_numpy = df[['Bob_X', 'Bob_Z', 'Eve_X', 'Eve_Z']].values
X_numpy /= 70.0
X = torch.tensor(X_numpy, dtype=torch.float32)

# Extract the Raw Action (512-D W matrix)
# Extract the Raw Action (512-D W matrix)
w_cols = [col for col in df.columns if col.startswith('W_Real_') or col.startswith('W_Imag_')]
W_raw_numpy = df[w_cols].values

# Scale the W matrix before tensor conversion
print("Loading scaler and scaling W matrices...")
scaler = joblib.load('xlmimo_w_scaler.pkl')
W_scaled_numpy = scaler.transform(W_raw_numpy)

# Convert the SCALED numpy array to a PyTorch tensor
W_tensor = torch.tensor(W_scaled_numpy, dtype=torch.float32)

print(f"Loaded {len(df)} samples. State shape: {X.shape}, Scaled W shape: {W_tensor.shape}")

# -------------------------------------------------------------------------
# 2. COMPRESS W INTO 16-D LATENT SPACE USING YOUR VAE
# -------------------------------------------------------------------------
print("Encoding 512-D W matrices into 16-D latent vectors...")

# Load the Model
vae = XLMIMO_VAE(input_dim=512, latent_dim=16)
vae.load_state_dict(torch.load('checkpoints/w_matrix_full_vae_epoch_1000.pth', map_location=torch.device('cpu')))
vae.eval() # Set to evaluation mode (turns off random dropout/batchnorm if you had them)
print("[OK] VAE Model weights loaded.")

with torch.no_grad():
    # We use the mean (mu) of the encoder as our deterministic optimal action
    mu, logvar = vae.encode(W_tensor)
    Y = mu  # Y is now shape [100000, 16]

epsilon = df[['Epsilon']].values
epsilon_tensor = torch.tensor(epsilon, dtype=torch.float32)
Y = torch.cat((Y, epsilon_tensor), dim=1)

# -------------------------------------------------------------------

# Create PyTorch DataLoader for fast batching
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# -------------------------------------------------------------------------
# 3. INITIALIZE FRESH SAC AGENT
# -------------------------------------------------------------------------
config = Config(yaml_path='config.yaml')
env = XLMIMOEnv(config=config)

model = SAC(
    'MlpPolicy',
    env,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
    device='cpu' # Using CPU since we determined earlier you don't have an NVIDIA GPU
)

# -------------------------------------------------------------------------
# 4. SUPERVISED PRE-TRAINING (BEHAVIOR CLONING)
# -------------------------------------------------------------------------
# Extract the Actor Network from SAC
actor_net = model.policy.actor
optimizer = torch.optim.Adam(actor_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 75

print("Starting Supervised Pre-Training (Behavior Cloning)...")
actor_net.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    
    for batch_x, batch_y in dataloader:
# Forward pass
        predicted_actions = actor_net(batch_x, deterministic=True)
        
        # Slice dim=1 (the features), keeping dim=0 (the batch) intact
        latent_action = predicted_actions[:, :16] * 3      # Shape: [batch_size, 16]
        psf = (predicted_actions[:, 16:] + 1) / 2          # Shape: [batch_size, 1] 
        
        # Combine back into [batch_size, 17]
        predicted_actions_scaled = torch.cat((latent_action, psf), dim=1)  
        
        # Calculate loss using the SCALED actions
        loss = loss_fn(predicted_actions_scaled, batch_y)
        
        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs} | MSE Loss: {avg_loss:.6f}")


# -------------------------------------------------------------------------
# 5. SAVE PRE-TRAINED BRAIN
# -------------------------------------------------------------------------
save_path = "sac_pretrained_brain.zip"
model.save(save_path)
print(f"\n[SUCCESS] Supervised Pre-training complete. Agent saved to {save_path}")
print("You can now load this file in your 8-core DQN_Train.py script for fine-tuning!")

# -------------------------------------------------------------------------
# 6. SANITY CHECK: PRINT MODEL OUTPUTS VS TARGETS
# -------------------------------------------------------------------------
model = SAC.load("sac_pretrained_brain.zip", device='cpu')
actor_net = model.policy.actor

print("\n--- SANITY CHECK: LATENT COMPARISON ---")
actor_net.eval() # Turn off training mode

with torch.no_grad():
    # Grab just one small batch of 3 from the dataloader
    test_x, test_y = next(iter(DataLoader(dataset, batch_size=3, shuffle=True)))
    
    # 1. Get raw predictions [-1, 1]
    raw_preds = actor_net(test_x, deterministic=True)
    
    # 2. Scale them exactly like we did in training
    scaled_latents = raw_preds[:, :16] * 3
    scaled_psf = (raw_preds[:, 16:] + 1) / 2
    scaled_preds = torch.cat((scaled_latents, scaled_psf), dim=1)
    
    for i in range(3):
        print(f"\nSample {i+1}:")
        print(f"Target Latent (First 5): {test_y[i, :5].numpy().round(3)}")
        print(f"Model  Latent (First 5): {scaled_preds[i, :5].numpy().round(3)}")
        print(f"Target Epsilon:        {test_y[i, 16].item():.3f}")
        print(f"Model  Epsilon:        {scaled_preds[i, 16].item():.3f}")
print("---------------------------------------")