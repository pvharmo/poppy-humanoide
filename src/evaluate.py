import gym
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG, TD3
from environment import RobotEnv
import os


def evaluate_model(model_path: str, scene_path: str, num_episodes: int = 10):
    """Evaluate a trained model"""

    # Create environment
    env = RobotEnv(scene_path=scene_path, headless=False)

    # Load model (detect algorithm from path)
    if "PPO" in model_path:
        model = PPO.load(model_path)
    elif "SAC" in model_path:
        model = SAC.load(model_path)
    elif "DDPG" in model_path:
        model = DDPG.load(model_path)
    elif "TD3" in model_path:
        model = TD3.load(model_path)
    else:
        raise ValueError("Could not detect algorithm from model path")

    # Evaluation
    total_rewards = []
    success_rate = 0

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            # Render if needed
            env.pr.step()

        total_rewards.append(episode_reward)

        # Check if episode was successful
        if info.get("episode", {}).get("success", False):
            success_rate += 1

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    # Print results
    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Success Rate: {(success_rate / num_episodes * 100):.1f}%")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    # Example usage
    model_path = "models/PPO_20231201_123456/best_model"  # Update with your model path
    scene_path = "path/to/your/scene.ttt"  # Update with your scene path

    evaluate_model(model_path, scene_path, num_episodes=5)
