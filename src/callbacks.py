from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    """

    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            if len(rewards) > 0:
                mean_reward = np.mean(rewards)
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - "
                        f"Last mean reward: {mean_reward:.2f}"
                    )

                # New best model, save it
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(os.path.join(self.save_path, "best_model"))

        return True


class ProgressBarCallback(BaseCallback):
    """A simple callback to display a progress bar during training"""

    def __init__(self, total_timesteps: int, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self) -> None:
        # Initialize progress bar
        try:
            from tqdm import tqdm

            self.progress_bar = tqdm(total=self.total_timesteps)
        except ImportError:
            print("tqdm not installed, progress bar disabled")

    def _on_step(self) -> bool:
        if self.progress_bar is not None:
            self.progress_bar.update(self.model.num_timesteps - self.progress_bar.n)
        return True

    def _on_training_end(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
