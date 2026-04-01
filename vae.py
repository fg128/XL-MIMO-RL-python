import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os

# -----------------------------------------
# 1. Custom Dataset Definition
# -----------------------------------------
class XLMIMODataset(Dataset):
    def __init__(self, csv_file):
        print(f"Loading dataset from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Drop everything except the W matrix
        cols_to_drop = ['Episode', 'Bob_X', 'Bob_Z', 'Eve_X', 'Eve_Z', 'Epsilon', 'Secrecy_Rate']
        existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df_w_matrix = df.drop(columns=existing_cols_to_drop)
        
        self.data = df_w_matrix.values.astype(np.float32)
        self.feature_dim = self.data.shape[1] # Should be exactly 512
        
        if self.feature_dim != 512:
            print(f"WARNING: Expected 512 features, got {self.feature_dim}. Check your CSV.")

        # Standardizing is mandatory for the MSE loss to care about small weight fluctuations
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        
        joblib.dump(self.scaler, 'xlmimo_w_scaler.pkl')
        print(f"Dataset loaded: {len(self.data)} samples. Scaler saved.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

# -----------------------------------------
# 2. VAE Architecture
# -----------------------------------------
class XLMIMO_VAE(nn.Module):
    def __init__(self, input_dim=512, latent_dim=16):
        super(XLMIMO_VAE, self).__init__()
        
        # ENCODER: Base feature extraction
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2)
        )
        
        # ENCODER: Split into Mean (mu) and Variance (logvar)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # DECODER: Reconstruct the W matrix from the latent vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, input_dim) # Linear output matching scaled data
        )

    def encode(self, x):
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """ The Reparameterization Trick: z = mu + std * epsilon """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        logvar = torch.clamp(logvar, -5, 5)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        print(f"DEBUG: mu={mu}, logvar={logvar}, z={z}") # Debugging line to check values

        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

# -----------------------------------------
# 3. VAE Loss Function (MSE + KL Divergence)
# -----------------------------------------
def vae_loss_function(recon_x, x, mu, logvar, beta=0.2, gamma=10.0):
    # 1. MSE Loss (Sum across features, Mean across batch)
    mse_loss = F.mse_loss(recon_x, x, reduction='none')
    mse_loss = torch.sum(mse_loss, dim=1).mean()
    
    # 2. KLD Loss (Sum across latents, Mean across batch)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # 3. Direction Loss (Mean across batch)
    cos_sim = F.cosine_similarity(recon_x, x, dim=1)
    dir_loss = torch.mean(1 - cos_sim) 
    
    total_loss = mse_loss + (beta * kld_loss) + (gamma * dir_loss)
    return total_loss, mse_loss, kld_loss, dir_loss

# -----------------------------------------
# 4. Training Loop
# -----------------------------------------
def train_vae(csv_filename="xlmimo_oracle_dataset.csv", batch_size=256, epochs=50, latent_dim=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = XLMIMODataset(csv_filename)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"DataLoader length: {len(dataloader)}")

    model = XLMIMO_VAE(input_dim=dataset.feature_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) 

    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        train_recon = 0.0
        train_kld = 0.0
        train_dir = 0.0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = model(batch)
            
            # Calculate loss
            loss, recon, kld, dir_loss = vae_loss_function(recon_batch, batch, mu, logvar, beta=0.4, gamma=30.0)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon.item()
            train_kld += kld.item()
            train_dir += dir_loss.item()
            
        avg_loss = train_loss / len(dataloader)
        avg_recon = train_recon / len(dataloader)
        avg_kld = train_kld / len(dataloader)
        avg_dir = train_dir / len(dataloader)

        print(f"Epoch [{epoch+1}/{epochs}] | Total: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KLD: {avg_kld:.4f} | Dir: {avg_dir:.4f} | mean {mu.mean().item():.4f} | std {mu.std().item():.4f}")

        if (epoch + 1) % 100 == 0:
            os.makedirs("checkpoints", exist_ok=True) # Creates folder if it doesn't exist
            
            full_path = f"checkpoints/w_matrix_full_vae_epoch_{epoch+1}.pth"
            dec_path = f"checkpoints/w_matrix_vae_decoder_epoch_{epoch+1}.pth"
            
            torch.save(model.state_dict(), full_path)
            torch.save(model.decoder.state_dict(), dec_path)
            print(f"--> Saved checkpoint: {full_path}")
        

    # Save full model and decoder
    torch.save(model.state_dict(), 'w_matrix_full_vae.pth')
    
    # We save just the decoder part. This is what the RL agent will use!
    torch.save(model.decoder.state_dict(), 'w_matrix_vae_decoder.pth')
    print("Training complete! Saved 'w_matrix_full_vae.pth' and 'w_matrix_vae_decoder.pth'")

if __name__ == "__main__":
    train_vae(csv_filename="xlmimo_oracle_dataset.csv", epochs=1000, latent_dim=16)