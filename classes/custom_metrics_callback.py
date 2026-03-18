import threading

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from classes.config import Config
from baselines.exhaustive_search import exhaustive_search
from baselines.mrt import mrt_no_an
from baselines.fixed_power_allocation import fixed_power_allocation
from functions.visualise import visualise

ACTION_NAMES = [
    "Stay",
    "Ang+1", "Ang+8",
    "Ang-1", "Ang-8",
    "Rng+1", "Rng+5",
    "Rng-1", "Rng-5",
    "PSF+.01", "PSF-.01",
    "PSF+.05", "PSF-.05",
]


class CustomMetricsCallback(BaseCallback):
    def __init__(self, config: Config, stop_event: "threading.Event | None" = None, verbose=0, action_log_interval=500):
        super().__init__(verbose)
        self.config = config
        self.stop_event = stop_event
        self.action_log_interval = action_log_interval
        self.action_counts = np.zeros(len(ACTION_NAMES), dtype=int)

        self._max_episode_sr = 0.0
        self._cached_bob_loc = None
        self._cached_eve_loc = None

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]

        if 'secrecy_rate' in info:
            sr = info['secrecy_rate']
            self.logger.record('metrics/secrecy_rate', sr)
            self.logger.record('metrics/dist_to_bob', info['dist_to_bob'])
            self.logger.record('metrics/dist_to_eve', info['dist_to_eve'])

            # Track best SR seen this episode
            if sr > self._max_episode_sr:
                self._max_episode_sr = sr

            # Cache locations every step (info preserved at terminal step before reset)
            if 'bob_loc' in info:
                self._cached_bob_loc = info['bob_loc']
                self._cached_eve_loc = info['eve_loc']

        # Track action taken this step
        action = int(self.locals['actions'][0])
        self.action_counts[action] += 1

        # Log action distribution every N steps
        if self.num_timesteps % self.action_log_interval == 0:
            total = self.action_counts.sum()
            fractions = self.action_counts / max(total, 1)

            for i, (name, frac) in enumerate(zip(ACTION_NAMES, fractions)):
                self.logger.record(f'actions/{i:02d}_{name}', frac)

            top3 = np.argsort(fractions)[::-1][:3]
            summary = "  ".join(
                f"{ACTION_NAMES[i]}={fractions[i]*100:.1f}%" for i in top3
            )
            print(f"[Step {self.num_timesteps}] Action dist (top 3): {summary}")

        # At episode end, log baseline comparisons
        dones = self.locals.get('dones', [False])
        if dones[0] and self._cached_bob_loc is not None and self._cached_eve_loc is not None:
            bob_loc = self._cached_bob_loc
            eve_loc = self._cached_eve_loc

            optimal_sr, best_beam_idx, best_psf = exhaustive_search(self.config, bob_loc, eve_loc)
            mrt_sr = mrt_no_an(self.config, bob_loc, eve_loc)
            fpa_srs = fixed_power_allocation(self.config, bob_loc, eve_loc, (0.5, 0.8))

            dqn_max_sr = self._max_episode_sr
            sr_efficiency = dqn_max_sr / optimal_sr if optimal_sr > 0 else 0.0

            self.logger.record('episode/optimal_sr', optimal_sr)
            self.logger.record('episode/mrt_sr', mrt_sr)
            self.logger.record('episode/fpa_0.5_sr', fpa_srs[0.5])
            self.logger.record('episode/fpa_0.8_sr', fpa_srs[0.8])
            self.logger.record('episode/dqn_max_sr', dqn_max_sr)
            self.logger.record('episode/sr_efficiency', sr_efficiency)

            # Reset for next episode
            self._max_episode_sr = 0.0

        if self.stop_event is not None and self.stop_event.is_set():
            return False

        return True
