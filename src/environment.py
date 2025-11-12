import gymnasium
from gymnasium import spaces
import numpy as np
from pyrep import PyRep
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from typing import Tuple, List


class RobotEnv(gymnasium.Env):
    """Custom robot environment for CoppeliaSim with Stable Baselines3"""

    def __init__(self, scene_path: str, headless: bool = False):
        super(RobotEnv, self).__init__()

        # Initialize PyRep (CoppeliaSim interface)
        self.pr = PyRep()
        self.pr.launch(scene_path, headless=headless)
        self.pr.start()

        # Get robot components (adjust names based on your scene)
        try:
            # Joints pour Poppy Humanoid - on ajustera les noms plus tard
            self.joint_names = [
                "l_hip_x", "l_hip_y", "l_hip_z", "l_knee", "l_ankle_y", "l_ankle_x",
                "r_hip_x", "r_hip_y", "r_hip_z", "r_knee", "r_ankle_y", "r_ankle_x",
                "abs_x", "abs_y", "abs_z",
                "l_shoulder_x", "l_shoulder_y", "l_arm_z",
                "r_shoulder_x", "r_shoulder_y", "r_arm_z"
            ]
            self.joints = [Joint(name) for name in self.joint_names]
            self.robot_base = Shape("poppy_base")  # Ajuster le nom
            self.target = Shape("target")
            self.vision_sensor = VisionSensor("vision_sensor")
        except Exception as e:
            print(f"Warning: Could not find all objects: {e}")
            self.joints = []
            self.robot_base = None
            self.target = None
            self.vision_sensor = None

        # Define action and observation space APRÈS avoir défini self.joints
        n_joints = len(self.joints)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32)

        # Observation: positions + vitesses + position robot + position target + distance
        obs_size = n_joints * 2 + 3 + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.episode_step = 0
        self.max_episode_steps = 500
        self.current_episode_reward = 0.0
        self.previous_distance = 0.0

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.pr.stop()
        self.pr.start()
        self.episode_step = 0
        self.current_episode_reward = 0.0

        # Reset joints avec positions initiales sûres
        initial_positions = self._get_initial_positions()
        for joint, pos in zip(self.joints, initial_positions):
            joint.set_joint_position(pos)
            joint.set_joint_target_velocity(0.0)

        # Randomize target position
        self._randomize_target()

        # Initialiser la distance précédente
        if self.robot_base and self.target:
            robot_pos = self.robot_base.get_position()
            target_pos = self.target.get_position()
            self.previous_distance = np.linalg.norm(np.array(robot_pos) - np.array(target_pos))

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one time step"""
        self.episode_step += 1

        # Apply action to joints avec sécurité réduite
        for i, joint in enumerate(self.joints):
            if i < len(action):
                joint.set_joint_target_velocity(action[i] * 1.0)  # Scale réduit

        # Step simulation
        self.pr.step()

        # Get observation, reward, done
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self.episode_step >= self.max_episode_steps or self._is_success() or self._has_fallen()

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

        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        try:
            # Joint positions and velocities (tous les joints)
            joint_positions = [joint.get_joint_position() for joint in self.joints]
            joint_velocities = [joint.get_joint_velocity() for joint in self.joints]

            # Robot and target positions
            robot_pos = self.robot_base.get_position() if self.robot_base else [0, 0, 0]
            target_pos = self.target.get_position() if self.target else [1, 1, 0]

            # Distance to target
            distance = np.linalg.norm(np.array(robot_pos) - np.array(target_pos))

            obs = np.array(
                joint_positions
                + joint_velocities
                + robot_pos
                + target_pos
                + [distance],
                dtype=np.float32,
            )

            return obs

        except Exception as e:
            print(f"Observation error: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def _compute_reward(self) -> float:
        """Compute reward based on current state"""
        try:
            robot_pos = self.robot_base.get_position() if self.robot_base else [0, 0, 0]
            target_pos = self.target.get_position() if self.target else [1, 1, 0]

            # Distance to target
            current_distance = np.linalg.norm(np.array(robot_pos) - np.array(target_pos))

            # Reward for getting closer to target
            distance_reward = (self.previous_distance - current_distance) * 10.0

            # Update previous distance
            self.previous_distance = current_distance

            # Bonus for being close to target
            success_bonus = 10.0 if current_distance < 0.2 else 0.0

            # Penalty for taking too long
            time_penalty = 0.01

            # Penalty for falling
            fall_penalty = 5.0 if self._has_fallen() else 0.0

            total_reward = distance_reward + success_bonus - time_penalty - fall_penalty

            return total_reward

        except Exception as e:
            print(f"Reward computation error: {e}")
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

    def _has_fallen(self) -> bool:
        """Check if robot has fallen"""
        try:
            if self.robot_base:
                position = self.robot_base.get_position()[2]  # Z height
                # Consider fallen if too low
                return position < 0.3
            return False
        except:
            return False

    def _get_initial_positions(self) -> List[float]:
        """Get safe initial joint positions for standing"""
        # Positions pour une posture debout stable
        return [
            # Left leg
            0.0, 0.0, 0.0, 0.7, -0.4, 0.0,
            # Right leg
            0.0, 0.0, 0.0, 0.7, -0.4, 0.0,
            # Torso and arms
            0.0, 0.0, 0.0,
            0.2, 0.0, 0.0,
            -0.2, 0.0, 0.0
        ]

    def _randomize_target(self):
        """Randomize target position"""
        try:
            import random

            new_pos = [
                random.uniform(0.5, 2.0),  # Devant le robot
                random.uniform(-1.0, 1.0),  # Sur les côtés
                0.05,  # Keep on ground
            ]
            if self.target:
                self.target.set_position(new_pos)
        except Exception as e:
            print(f"Target randomization error: {e}")

    def render(self, mode="human"):
        """Render the environment"""
        # Rendering is handled by CoppeliaSim
        pass

    def close(self):
        """Clean up"""
        self.pr.stop()
        self.pr.shutdown()