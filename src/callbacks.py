from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

class ProgressBarCallback(BaseCallback):
    """A simple callback to display a progress bar during training"""

    def __init__(self, total_timesteps: int, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(total=self.total_timesteps)

    def _on_step(self) -> bool:
        if self.progress_bar is not None:
            self.progress_bar.update(self.model.num_timesteps - self.progress_bar.n)
        return True

    def _on_training_end(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
