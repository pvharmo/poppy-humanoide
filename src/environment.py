from typing import List, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from poppy_humanoid import PoppyHumanoid
from pypot.vrep.io import VrepIO
from pypot.vrep.remoteApiBindings.vrep import simxSynchronous, simxSynchronousTrigger


class RobotEnv(gymnasium.Env):
    """Custom robot environment for CoppeliaSim with Stable Baselines3"""

    def __init__(self, headless: bool = False):
        super(RobotEnv, self).__init__()

        # Initialize PyRep (CoppeliaSim interface)
        self.vrep = VrepIO("127.0.0.1", 19997)

        self.poppy = PoppyHumanoid(simulator="vrep", shared_vrep_io=self.vrep)
        simxSynchronous(self.vrep.client_id, True)

        # Define action and observation space
        # Continuous action space for joint control
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(25,), dtype=np.float32
        )

        # Observation space: joint positions, velocities, target position, distance
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(34,), dtype=np.float32
        )

        # Get robot components (adjust names based on your scene)

        self.motors = [m.present_position for m in self.poppy.motors]
        print(f"{len(self.motors)} Motors initialized")
        self.head_position = self.poppy.get_object_position("head_visual")
        self.head_target = [10, 0, self.head_position[2]]

        self.episode_step = 0
        self.max_episode_steps = 500
        self.current_episode_reward = 0.0
        self.previous_distance = 0.0

        print("Environment initialized")

    def reset(self, seed=None) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state"""
        print("Resetting environment")
        # self.poppy.reset_simulation()
        print("Simulation reset")

        self.motors = [m.present_position for m in self.poppy.motors]
        self.head_position = self.poppy.get_object_position("head_visual")
        self.head_target = (10, 0, self.head_position[2])

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step"""
        self.episode_step += 1
        print(f"running step {self.episode_step}")

        # Apply action to joints
        for i, motor in enumerate(self.poppy.motors):
            if i < len(action):
                motor.goto_position(action[i], 0.1)  # Scale action

        # Step simulation
        simxSynchronousTrigger(self.vrep.client_id)

        # Get observation, reward, done
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._is_success() or self._has_fallen()

        self.current_episode_reward += reward

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
        head_position = np.array(self.poppy.get_object_position("head_visual"))
        torso_position = np.array(self.poppy.get_object_position("spine_visual"))

        return head_position[2] - torso_position[2] < 0.05

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Joint positions and velocities
        joint_positions = [m.present_position for m in self.poppy.motors]
        # joint_velocities = [joint.get_joint_velocity() for joint in self.joints[:4]]

        # Robot and target positions
        head_position = self.poppy.get_object_position("head_visual")
        torso_position = self.poppy.get_object_position("spine_visual")

        target = list(self.head_target)

        obs = np.array(
            joint_positions + head_position + torso_position + target,
            dtype=np.float32,
        )

        return obs

    def _compute_reward(self) -> float:
        """Compute reward based on current state"""

        head_position = np.array(self.poppy.get_object_position("head_visual"))
        torso_position = np.array(self.poppy.get_object_position("spine_visual"))

        head_torso_diff = head_position - torso_position
        fall_penalty = abs(head_torso_diff[0]) + abs(head_torso_diff[1])

        # Distance to target
        distance = np.linalg.norm(np.array(head_position) - np.array(self.head_target))

        # Reward for getting closer to target
        base_reward = 10 - distance - fall_penalty

        # Penalty for taking too long
        base_reward -= self.episode_step * 0.001

        return base_reward

    def _is_success(self) -> bool:
        """Check if task is completed"""
        head_position = np.array(self.poppy.get_object_position("head_visual"))

        # Distance to target
        distance = np.linalg.norm(np.array(head_position) - np.array(self.head_target))
        return distance < 0.5

    # def _randomize_target(self):
    #     """Randomize target position"""
    #     try:
    #         import random

    #         new_pos = [
    #             random.uniform(-0.5, 0.5),
    #             random.uniform(-0.5, 0.5),
    #             0.05,  # Keep on ground
    #         ]
    #         if self.target:
    #             self.target.set_position(new_pos)
    #     except:
    #         pass

    def render(self, mode="human"):
        """Render the environment"""
        # Rendering is handled by CoppeliaSim
        pass

    def close(self):
        """Clean up"""
        # self.pr.shutdown()
