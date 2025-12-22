from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, ConvertCallback
import pandas as pd
import json, os
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

class LogInfoEveryNTimesteps(EveryNTimesteps):
    """
    Log data every ``n_steps`` timesteps

    :param n_steps: Number of timesteps between two trigger.
    """

    def __init__(self, n_steps: int):
        super().__init__(n_steps, callback=ConvertCallback(self._log_data))

    def _log_data(self, _locals: dict[str, Any], _globals: dict[str, Any]) -> bool:
        infos = _locals.get("infos")
        print(type(infos), len(infos))

        survive_mean = safe_mean([info["reward_survive"] for info in infos])
        self.model.logger.record("reward/survive", survive_mean)
        forward_mean = safe_mean([info["reward_forward"] for info in infos])
        self.model.logger.record("reward/forward", forward_mean)
        ctrl_mean = safe_mean([info["reward_ctrl"] for info in infos])
        self.model.logger.record("reward/control", ctrl_mean)
        contact_mean = safe_mean([info["reward_contact"] for info in infos])
        self.model.logger.record("reward/contact", contact_mean)

        return True


class ReplaySaveCallback(BaseCallback):

    def __init__(self, path: str, model, env,verbose: int = 0):
        super().__init__(verbose)
        self.path=path
        self.model=model
        self.env=env
        self.dataframe=self.env.dataframe
 
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:

        pass

    def _on_step(self) -> bool:
        self.dataframe.to_csv(f'{self.path}/replay.csv', index=False)
            

        return True

    def _on_rollout_end(self) -> None:

        pass

    def _on_training_end(self) -> None:

        pass