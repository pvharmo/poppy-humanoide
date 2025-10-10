import gym
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from environment import RobotEnv
from callbacks import SaveOnBestTrainingRewardCallback, ProgressBarCallback
import os
from datetime import datetime


def create_env(scene_path: str, headless: bool = False):
    """Create and wrap the environment"""
    env = RobotEnv(scene_path=scene_path, headless=headless)
    env = Monitor(env)  # Add monitoring for rewards
    return env


def train():
    # Configuration
    SCENE_PATH = "path/to/your/scene.ttt"  # Update this path
    ALGORITHM = "PPO"  # Choose from: PPO, SAC, DDPG, TD3
    TOTAL_TIMESTEPS = 100000
    SAVE_FREQ = 10000
    HEADLESS = False

    # Create environment
    env = create_env(SCENE_PATH, HEADLESS)

    # Check if environment follows gym interface
    check_env(env)

    # Create log directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{ALGORITHM}_{timestamp}"
    model_dir = f"models/{ALGORITHM}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Choose algorithm
    if ALGORITHM == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
    elif ALGORITHM == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
        )
    elif ALGORITHM == "DDPG":
        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=1e-3,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
        )
    elif ALGORITHM == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=1e-3,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
        )
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    # Callbacks
    callbacks = [
        SaveOnBestTrainingRewardCallback(check_freq=SAVE_FREQ, save_path=model_dir),
        ProgressBarCallback(total_timesteps=TOTAL_TIMESTEPS),
    ]

    print(f"Starting {ALGORITHM} training for {TOTAL_TIMESTEPS} timesteps...")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")

    # Train the model
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name=f"{ALGORITHM}_run",
        log_interval=1,
    )

    # Save final model
    model.save(os.path.join(model_dir, "final_model"))

    print("Training completed!")
    print(f"Final model saved to: {os.path.join(model_dir, 'final_model')}")

    env.close()


if __name__ == "__main__":
    train()
