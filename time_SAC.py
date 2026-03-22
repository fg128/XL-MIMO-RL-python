import time
import numpy as np
from stable_baselines3 import SAC

def time_sac_inference(model_path="checkpoints/SAC_10_77924.zip"):
    # 1. Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        model = SAC.load(model_path, device='cpu')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Create a dummy observation
    # Your observation space is exactly 7 floats:
    # [r_beam, theta_beam, psf, delta_r_bob, delta_theta_bob, delta_r_eve, delta_theta_eve]
    dummy_obs = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.8, 0.8], dtype=np.float32)

    # 3. Warm-up run
    # PyTorch/SB3 always takes longer on the very first predict() due to memory allocation.
    # We run it once un-timed to "warm up" the network.
    _ = model.predict(dummy_obs, deterministic=True)

    # 4. Timing Loop
    num_iterations = 10_000
    print(f"Running {num_iterations} iterations to find average inference time...")

    start_time = time.perf_counter()

    for _ in range(num_iterations):
        # deterministic=True ensures it exploits the learned policy rather than exploring
        action, _states = model.predict(dummy_obs, deterministic=True)

    end_time = time.perf_counter()

    # 5. Calculate Results
    total_time = end_time - start_time
    avg_time_per_step_sec = total_time / num_iterations
    avg_time_per_step_ms = avg_time_per_step_sec * 1000
    avg_time_per_step_us = avg_time_per_step_sec * 1_000_000

    print("-" * 50)
    print(f"Total time for {num_iterations} runs: {total_time:.4f} seconds")
    print(f"Average Inference Time per step: {avg_time_per_step_ms:.4f} milliseconds")
    print(f"                                 ({avg_time_per_step_us:.2f} microseconds)")
    print("-" * 50)

if __name__ == "__main__":
    time_sac_inference()
