import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

import functions.step_function as step_fn
import baselines.mrt as mrt_baseline
import baselines.fpa as fpa_baseline

class CustomMetricsCallback(BaseCallback):
    def __init__(self, config, verbose=0):
        super().__init__(verbose)
        self.config = config

        # --- NEW: Buffers for the 100-episode plot ---
        self.episode_count = 0
        self.current_episode_phis = []  # Tracks phi during a single episode
        self.plot_diffs = []            # Tracks diffs over 100 episodes
        self.plot_phis = []             # Tracks mean phis over 100 episodes

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]

        if 'secrecy_rate' in info:
            self.logger.record('metrics/secrecy_rate', info['secrecy_rate'])
            self.logger.record('metrics/dist_to_bob', info['dist_to_bob'])
            self.logger.record('metrics/dist_to_eve', info['dist_to_eve'])
            self.logger.record('metrics/phi', info['phi'])

            bob_loc = info['bob_loc']
            eve_loc = info['eve_loc']
            diff = np.linalg.norm(np.array(bob_loc) - np.array(eve_loc))
            self.logger.record('metrics/bob_eve_distance', diff)

            # Save step phi for averaging at the end of the episode
            self.current_episode_phis.append(info['phi'])


        # At episode end, compute baselines and generate plots
        done = self.locals['dones'][0]
        if done and 'bob_loc' in info:
            bob_loc = info['bob_loc']
            eve_loc = info['eve_loc']

            # Calculate final diff and mean phi for the episode that just finished
            diff = np.linalg.norm(np.array(bob_loc) - np.array(eve_loc))
            mean_phi = np.mean(self.current_episode_phis) if self.current_episode_phis else 0.0

            # Store them in our plot buffers
            self.plot_diffs.append(diff)
            self.plot_phis.append(mean_phi)
            self.current_episode_phis = [] # Reset for the next episode
            self.episode_count += 1

            # Baselines
            mrt_sr    = mrt_baseline.compute_secrecy_rate(self.config, bob_loc, eve_loc)
            fpa_90_sr = fpa_baseline.compute_secrecy_rate(self.config, bob_loc, eve_loc, phi=0.9)
            fpa_80_sr = fpa_baseline.compute_secrecy_rate(self.config, bob_loc, eve_loc, phi=0.8)
            fpa_70_sr = fpa_baseline.compute_secrecy_rate(self.config, bob_loc, eve_loc, phi=0.7)

            self.logger.record('metrics/mrt_secrecy_rate', mrt_sr)
            self.logger.record('metrics/fpa_90_secrecy_rate', fpa_90_sr)
            self.logger.record('metrics/fpa_80_secrecy_rate', fpa_80_sr)
            self.logger.record('metrics/fpa_70_secrecy_rate', fpa_70_sr)

            # --- NEW: Plot every 100 episodes ---
            if self.episode_count % 50 == 0:
                fig, ax = plt.subplots(figsize=(6, 4))

                # Create a scatter plot of Distance vs Phi
                ax.scatter(self.plot_diffs, self.plot_phis, alpha=0.6, color='tab:blue', edgecolors='k')
                ax.set_xlabel("Distance between Bob and Eve (m)")
                ax.set_ylabel("Agent's Chosen Phi")
                ax.set_title(f"SAC Phi Output vs Distance (Episodes {self.episode_count-100} to {self.episode_count})")
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.0) # Assuming phi is between 0 and 1
                ax.set_xlim(0, 1.0)

                # Push the figure to TensorBoard
                self.logger.record("plots/phi_vs_distance", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))

                # Clear buffers for the next 100 episodes
                self.plot_diffs = []
                self.plot_phis = []

        if step_fn._save_checkpoint:
            step_fn._save_checkpoint = False
            path = f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(path)
            print(f"\n[Checkpoint saved → {path}]")

        return True
