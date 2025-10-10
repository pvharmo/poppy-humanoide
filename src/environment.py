import gym
from gym import spaces
import numpy as np
from pyrep import PyRep
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from typing import Tuple, List

class RobotEnv(gym.Env):
    """Custom robot environment for CoppeliaSim with Stable Baselines3"""

    def __init__(self, scene_path: str, headless: bool = False):
        super(RobotEnv, self).__init__()

        # Initialize PyRep (CoppeliaSim interface)
        self.pr = PyRep()
        self.pr.launch(scene_path, headless=headless)
        self.pr.start()

        # Define action and observation space
        # Continuous action space for joint control
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space: joint positions, velocities, target position, distance
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        # Get robot components (adjust names based on your scene)
        try:
            self.joints = [Joint(f'joint_{i}') for i in range(4)]
            self.robot_base = Shape('robot_base')
            self.target = Shape('target')
            self.vision_sensor = VisionSensor('vision_sensor')
        except:
            print("Warning: Could not find all objects. Using dummy objects.")
            self.joints = []
            self.robot_base = None
            self.target = None
            self.vision_sensor = None

        self.episode_step = 0
        self.max_episode_steps = 500
        self.current_episode_reward = 0.0

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.pr.stop()
        self.pr.start()
        self.episode_step = 0
        self.current_episode_reward = 0.0

        # Reset joints
        for joint in self.joints:
            joint.set_joint_position(0.0)
            joint.set_joint_target_velocity(0.0)

        # Randomize target position
        self._randomize_target()

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one time step"""
        self.episode_step += 1

        # Apply action to joints
        for i, joint in enumerate(self.joints):
            if i < len(action):
                joint.set_joint_target_velocity(action[i] * 2.0)  # Scale action

        # Step simulation
        self.pr.step()

        # Get observation, reward, done
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self.episode_step >= self.max_episode_steps or self._is_success()

        self.current_episode_reward += reward

        # Additional info
        info = {
            'episode': {
                'r': self.current_episode_reward,
                'l': self.episode_step,
                'success': self._is_success()
            }
        }

        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        try:
            # Joint positions and velocities
            joint_positions = [joint.get_joint_position() for joint in self.joints[:4]]
            joint_velocities = [joint.get_joint_velocity() for joint in self.joints[:4]]

            # Robot and target positions
            robot_pos = self.robot_base.get_position() if self.robot_base else [0, 0, 0]
            target_pos = self.target.get_position() if self.target else [1, 1, 0]

            # Distance to target
            distance = np.linalg.norm(np.array(robot_pos) - np.array(target_pos))

            obs = np.array(
                joint_positions +
                joint_velocities +
                robot_pos +
                target_pos +
                [distance],
                dtype=np.float32
            )

            # Ensure correct shape
            if len(obs) < 11:
                obs = np.pad(obs, (0, 11 - len(obs)), mode='constant')

            return obs[:11]  # Ensure exactly 11 dimensions

        except Exception as e:
            print(f"Observation error: {e}")
            return np.zeros(11, dtype=np.float32)

    def _compute_reward(self) -> float:
        """Compute reward based on current state"""
        try:
            robot_pos = self.robot_base.get_position() if self.robot_base else [0, 0, 0]
            target_pos = self.target.get_position() if self.target else [1, 1, 0]

            # Distance to target
            distance = np.linalg.norm(np.array(robot_pos) - np.array(target_pos))

            # Reward for getting closer to target
            base_reward = -distance * 0.1

            # Bonus for being close to target
            if distance < 0.1:
                base_reward += 10.0

            # Penalty for taking too long
            base_reward -= 0.01

            return base_reward

        except:
            return 0.0

    def _is_success(self) -> bool:
        """Check if task is completed"""
        try:
            robot_pos = self.robot_base.get_position() if self.robot_base else [0, 0, 0]
            target_pos = self.target.get_position() if self.target else [1, 1, 0]
            distance = np.linalg.norm(np.array(robot_pos) - np.array(target_pos))
            return distance < 0.1
        except:
            return False

    def _randomize_target(self):
        """Randomize target position"""
        try:
            import random
            new_pos = [
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                0.05  # Keep on ground
            ]
            if self.target:
                self.target.set_position(new_pos)
        except:
            pass

    def render(self, mode='human'):
        """Render the environment"""
        # Rendering is handled by CoppeliaSim
        pass

    def close(self):
        """Clean up"""
        self.pr.stop()
        self.pr.shutdown()
