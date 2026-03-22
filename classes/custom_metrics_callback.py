import functions.step_function as step_fn
from stable_baselines3.common.callbacks import BaseCallback

import baselines.mrt as mrt_baseline
import baselines.fpa as fpa_baseline

class CustomMetricsCallback(BaseCallback):
    def __init__(self, config, verbose=0):
        super().__init__(verbose)
        self.config = config

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]

        if 'secrecy_rate' in info:
            self.logger.record('metrics/secrecy_rate', info['secrecy_rate'])
            self.logger.record('metrics/dist_to_bob', info['dist_to_bob'])
            self.logger.record('metrics/dist_to_eve', info['dist_to_eve'])

        # At episode end, compute and log MRT and FPA baseline secrecy rates
        done = self.locals['dones'][0]
        if done and 'bob_loc' in info:
            bob_loc = info['bob_loc']
            eve_loc = info['eve_loc']

            mrt_sr    = mrt_baseline.compute_secrecy_rate(self.config, bob_loc, eve_loc)
            fpa_90_sr = fpa_baseline.compute_secrecy_rate(self.config, bob_loc, eve_loc, phi=0.9)
            fpa_80_sr = fpa_baseline.compute_secrecy_rate(self.config, bob_loc, eve_loc, phi=0.8)
            fpa_70_sr = fpa_baseline.compute_secrecy_rate(self.config, bob_loc, eve_loc, phi=0.7)

            self.logger.record('metrics/mrt_secrecy_rate', mrt_sr)
            self.logger.record('metrics/fpa_90_secrecy_rate', fpa_90_sr)
            self.logger.record('metrics/fpa_80_secrecy_rate', fpa_80_sr)
            self.logger.record('metrics/fpa_70_secrecy_rate', fpa_70_sr)

        if step_fn._save_checkpoint:
            step_fn._save_checkpoint = False
            path = f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(path)
            print(f"\n[Checkpoint saved → {path}]")

        return True
