import os
from time import sleep

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import torch.nn as nn
import paths
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print


from callbacks import ProgressBarCallback
from mujoco_env import PoppyEnv

os.system('cls' if os.name == 'nt' else 'clear')

# --- 1. HYPERPARAMETERS & CONFIG ---
N_STEPS = 4096
ENV_NUM = 8
SEED = 42
ITERATIONS = 100

TRAIN_STEPS = N_STEPS * ENV_NUM * ITERATIONS

print(f"Training for {TRAIN_STEPS:_} steps".replace("_", " "))

env = PoppyEnv(paths.scene_path, render_mode="rgb_array")

# --- 2. VECTORIZED & NORMALIZED ENVIRONMENTS ---
# Wrap training envs with normalization
env = make_vec_env(
    PoppyEnv,
    n_envs=ENV_NUM,
    env_kwargs={"model_path": paths.scene_path},
    seed=SEED,
)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)

eval_env = make_vec_env(
    PoppyEnv,
    n_envs=1,
    env_kwargs={"model_path": paths.scene_path, "render_mode": "rgb_array"},
    seed=SEED + 100,
)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=paths.chkpts_path)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=paths.exports_path,
    log_path=paths.logs_path,
    eval_freq=10000,
    n_eval_episodes=2000,
    deterministic=True,
    render=True,
)
callbacks = CallbackList([
    checkpoint_callback,
    ProgressBarCallback(total_timesteps=TRAIN_STEPS)
])

# --- 4. OPTIMIZED PPO MODEL ---
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=paths.logs_path,
    seed=SEED,

    # Core hyperparameters
    learning_rate=lambda f: f * 3e-4,  # Anneal LR
    n_steps=4096,
    batch_size=128,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.98,

    # Exploration & stability
    clip_range=lambda f: f * 0.2,  # Anneal clipping
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,

    # Network architecture
    policy_kwargs=dict(
        net_arch=[512, 512],
        activation_fn=nn.Tanh,
    ),
)

# --- 5. TRAINING ---
command = "tensorboard --logdir " + paths.logs_path
syntax = Syntax(command, "bash", theme="monokai", background_color="default")
panel = Panel(
    syntax,
    title="[bold green]Run TensorBoard with the following command[/]",
    title_align="left",
    border_style="green",
)
print(panel)

sleep(2)

model.learn(total_timesteps=TRAIN_STEPS, tb_log_name="PoppyRun", callback=callbacks)

# --- 6. SAVE EVERYTHING ---
print("Saving model and normalization stats...")

os.makedirs(paths.exports_path, exist_ok=True)
model.save(paths.exports_path + "/ppo_poppy_final")
env.save(paths.exports_path + "/vecnormalize.pkl")
eval_env.save(paths.exports_path + "/eval_vecnormalize.pkl")

env.close()
eval_env.close()
