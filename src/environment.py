from typing import Tuple

import gymnasium
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from gymnasium import spaces


class RobotEnv(gymnasium.Env):
    def __init__(
        self,
        headless: bool = False,
        render_mode="rgb_array",
        camera_id: int | None = None,
        camera_name: str | None = None,
    ):
        self.render_mode = render_mode
        super(RobotEnv, self).__init__()

        xml = open("../scene/scene.xml", "r").read()
        self.scene = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.scene)
        self.positions = []

        # Define action and observation space
        # Continuous action space for joint control
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.data.ctrl),), dtype=np.float32
        )

        # Observation space: joint positions, velocities, target position, distance
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.data.qpos) + len(self.data.qvel) + 3,),
            dtype=np.float32,
        )

        self.target = np.array([10, 1, 0])

        self.episode_step = 0
        self.max_episode_steps = 500
        self.current_episode_reward = 0.0
        self.previous_distance = 0.0

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.scene,
            self.data,
            default_camera_config,
            self.width,
            self.height,
            max_geom,
            camera_id,
            camera_name,
            visual_options,
        )

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state"""
        print("reseting environment")
        xml = open("../scene/scene.xml", "r").read()
        self.scene = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.scene)
        self.episode_step = 0

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step"""
        self.episode_step += 1

        # Apply action to joints
        self.data.ctrl = action

        # Step simulation
        mujoco.mj_step(self.scene, self.data)
        self.positions += [self.data.qpos[:3].copy()]

        # Get observation, reward, done
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._is_success() or self._has_fallen()

        self.current_episode_reward += reward

        if self.episode_step % 10000 == 0:
            positions = np.array(self.positions)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
            print(positions)
            ax.plot(positions[:, 2], c="r")
            print(np.min(positions[:, 2]))
            plt.savefig("positions.png")

        # Additional info
        info = {
            "episode": {
                "r": self.current_episode_reward,
                "l": self.episode_step,
                "success": self._is_success(),
                "fallen": self._has_fallen(),
            }
        }

        return obs, reward, done, False, info

    def _has_fallen(self):
        return self.data.qpos[2] < -0.1

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        return np.concatenate([self.data.qpos, self.data.qvel, self.target])

    def _compute_reward(self) -> float:
        """Compute reward based on current state"""

        head_position = self.scene.body("head").pos
        torso_position = self.scene.body("chest").pos

        head_torso_diff = head_position - torso_position
        fall_penalty = abs(head_torso_diff[0]) + abs(head_torso_diff[1])

        # Distance to target
        distance = np.linalg.norm(head_position - self.target)

        # Reward for getting closer to target
        base_reward = 10 - distance - fall_penalty

        # Penalty for taking too long
        # base_reward -= self.episode_step * 0.00001

        return base_reward

    def _is_success(self) -> bool:
        """Check if task is completed"""
        head_position = self.scene.body("head").pos

        # Distance to target
        distance = np.linalg.norm(head_position - self.target)
        return bool(distance < 0.5)

    def render(self, mode="human"):
        """Render the environment"""
        pass

    def close(self):
        positions = np.array(self.positions)
        plt.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="r")
        plt.show()
        """Clean up"""
