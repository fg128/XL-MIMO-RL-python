import csv
import os
import numpy as np
import traceback

from tqdm import tqdm
from SCA_GCC_convex import XLMIMO_System


def generate_oracle_dataset(num_episodes=100000, filename="xlmimo_oracle_dataset.csv"):
    N_bx = 256
    
    # 1. Open CSV in 'append' mode so we never overwrite existing data
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # 2. Build the massive header if the file is brand new
        if not file_exists:
            header = ['Episode', 'Bob_X', 'Bob_Z', 'Eve_X', 'Eve_Z', 'Epsilon', 'Secrecy_Rate']
            for i in range(N_bx):
                header.extend([f'W_Real_{i}', f'W_Imag_{i}'])
            writer.writerow(header)
            print(f"Created new dataset file: {filename}")
        
        print(f"Starting Data Generation for {num_episodes} episodes...")
        print("Data is saved to disk continuously. You can stop the script safely at any time (Ctrl+C).")
        print("-" * 65)

        for ep in tqdm(range(num_episodes), desc="Generating Episodes"):
            # 3. Randomize User Locations (Based on your grid limits)
            # X coordinate: -70m to 70m | Z (Depth) coordinate: Bob: 0m to 70m, Eve: 0m to 70m
            bob_x = np.random.uniform(-70, 70)
            bob_z = np.random.uniform(0, 70)
            
            eve_x = np.random.uniform(-70, 70)
            eve_z = np.random.uniform(0, 70)
                
            bob_loc = np.array([bob_x, 0, bob_z])
            eve_loc = np.array([eve_x, 0, eve_z])
            
            try:
                # 4. Initialize Environment and Run Optimization
                xl_mimo = XLMIMO_System(N_bx=N_bx, bob_loc=bob_loc, eve_loc=eve_loc)
                W_t, epsilon_t, secrecy_rate = xl_mimo.run_alternating_optimization()
                # print(f"\n[Episode {ep+1}/{num_episodes}] Bob: ({bob_x:.1f}, {bob_z:.1f}) | Eve: ({eve_x:.1f}, {eve_z:.1f}) | Epsilon: {epsilon_t:.3f} | Secrecy Rate: {secrecy_rate:.4f} bps/Hz")
                # If SCA totally fails and returns None, skip this episode
                if W_t is None:
                    print("Optimization returned None. Skipping to next episode.")
                    continue
                
                # 5. Flatten the complex matrix
                # W_t shape is (256, 1). Flatten turns it to a 1D array of 256 complex numbers.
                W_flat = W_t.flatten()
                
                # 6. Construct the row
                row = [ep, bob_x, bob_z, eve_x, eve_z, epsilon_t, secrecy_rate]
                for w in W_flat:
                    row.extend([w.real, w.imag])
                    
                # 7. Write and Flush immediately to hard drive
                writer.writerow(row)
                f.flush()
                
            except Exception as e:
                print(f"CRITICAL ERROR in Episode {ep+1}: {e}")
                traceback.print_exc()
                print("Skipping to next episode to keep the loop alive...")

if __name__ == "__main__":
    # Start the data collection! 
    # (Maybe change this to 10 for your first test run to ensure it writes to the CSV correctly)
    generate_oracle_dataset(num_episodes=100_000, filename="xlmimo_oracle_dataset.csv")