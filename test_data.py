import pandas as pd
import numpy as np

from SCA_GCC_convex import XLMIMO_System

df = pd.read_csv("xlmimo_oracle_dataset.csv", nrows=1000)

# Select first epsidoe with x greater than 30 for Bob
df = df[df['Bob_X'] < -30]
df = df.iloc[10:11] # Just take the first one that meets the criteria

bob_x = df['Bob_X'].values
bob_z = df['Bob_Z'].values
eve_x = df['Eve_X'].values
eve_z = df['Eve_Z'].values

# Print range of Bob and Eve locations in the dataset
print(f"Bob X range: {df['Bob_X'].min():.1f} to {df['Bob_X'].max():.1f} | Bob Z range: {df['Bob_Z'].min():.1f} to {df['Bob_Z'].max():.1f}")
print(f"Eve X range: {df['Eve_X'].min():.1f} to {df['Eve_X'].max():.1f} | Eve Z range: {df['Eve_Z'].min():.1f} to {df['Eve_Z'].max():.1f}")

print(f"Epsisode {df['Episode'].values[0]} | Bob: ({bob_x[0]:.1f}, {bob_z[0]:.1f}) | Eve: ({eve_x[0]:.1f}, {eve_z[0]:.1f})")
print(df.columns)

xl_mimo = XLMIMO_System(N_bx=256, N_bz=1, K=1, E=1, bob_loc=np.array([bob_x[0], 0, bob_z[0]]), eve_loc=np.array([eve_x[0], 0, eve_z[0]]))
W_t, epsilon_t, secrecy_rate = xl_mimo.run_alternating_optimization()

W_loaded = df[[f'W_Real_{i}' for i in range(256)] + [f'W_Imag_{i}' for i in range(256)]].values.astype(np.float32)
W_loaded_complex = W_loaded[:, :256] + 1j * W_loaded[:, 256:]
W_loaded_complex = W_loaded_complex.reshape(256, 1)

xl_mimo.visualize_beamfocusing(W_loaded_complex, epsilon_t, lue_coords=np.array([(xl_mimo.bob_theta, xl_mimo.bob_r)]), eue_coords=np.array([(xl_mimo.eve_theta, xl_mimo.eve_r)]))
print(f"Final Achieved Secrecy Rate: {secrecy_rate:.4f} bps/Hz")
