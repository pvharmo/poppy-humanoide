# Training configuration
TRAINING_CONFIG = {
    "scene_path": "path/to/your/scene.ttt",
    "algorithm": "PPO",  # PPO, SAC, DDPG, TD3
    "total_timesteps": 100000,
    "learning_rate": 3e-4,
    "batch_size": 64,
    "buffer_size": 1000000,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_kwargs": {
        "net_arch": [64, 64]  # Network architecture
    },
}

# Environment configuration
ENV_CONFIG = {
    "max_episode_steps": 500,
    "reward_scale": 1.0,
    "success_threshold": 0.1,  # Distance threshold for success
}
