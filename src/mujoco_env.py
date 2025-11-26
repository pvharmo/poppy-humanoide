from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class PoppyEnv(HumanoidEnv):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, healthy_z_range=(0.3, 0.6), reset_noise_scale=0, **kwargs)

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }

        return reward, reward_info
