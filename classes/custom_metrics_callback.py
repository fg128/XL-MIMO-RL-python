from stable_baselines3.common.callbacks import BaseCallback

class CustomMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # self.locals['infos'] contains the info dicts from the environment
        info = self.locals['infos'][0]

        # Check if our custom metric is in the info dict
        if 'secrecy_rate' in info:
            # Log to TensorBoard under a custom category called 'metrics'
            self.logger.record('metrics/secrecy_rate', info['secrecy_rate'])
            self.logger.record('metrics/dist_to_bob', info['dist_to_bob'])
            self.logger.record('metrics/dist_to_eve', info['dist_to_eve'])

        return True
