import os
import subprocess
import webbrowser

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    LogEveryNTimesteps,
)

from mujoco_env import PoppyEnv

os.makedirs("./a2c_poppy_tensorboard/logs", exist_ok=True)
os.makedirs("./a2c_poppy_tensorboard/chkpts", exist_ok=True)

env = PoppyEnv("../scene/scene.xml", render_mode="rgb_array")

record_video = False

if record_video:
    env = RecordVideo(
        env,
        video_folder="poppy_videos",  # Folder to save videos
        name_prefix="eval",  # Prefix for video filenames
        episode_trigger=lambda x: True,  # Record every episode
    )

checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path="./a2c_poppy_tensorboard/chkpts"
)
log_callback = LogEveryNTimesteps(n_steps=5)
callbacks = CallbackList([checkpoint_callback, log_callback])

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./a2c_poppy_tensorboard/logs")
print("Model initialized")
subprocess.Popen("tensorboard --logdir ./a2c_poppy_tensorboard/", shell=True)
webbrowser.open("http://localhost:6006")
model.learn(total_timesteps=50000, tb_log_name="Run", callback=callbacks)
os.makedirs("./a2c_poppy_tensorboard/exports", exist_ok=True)
model.save("./a2c_poppy_tensorboard/exports/a2c_poppy")
