import os
import subprocess

from selenium import webdriver
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    LogEveryNTimesteps,
)

from environment import RobotEnv

os.makedirs("./a2c_poppy_tensorboard/logs", exist_ok=True)
os.makedirs("./a2c_poppy_tensorboard/chkpts", exist_ok=True)

driver = webdriver.Firefox()
driver.get("localhost:6006")

print("Log folder created")
env = RobotEnv()
print("Environment initialized")

checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path="./a2c_poppy_tensorboard/chkpts"
)
log_callback = LogEveryNTimesteps(n_steps=1)
callbacks = CallbackList([checkpoint_callback, log_callback])

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./a2c_poppy_tensorboard/logs")
print("Model initialized")
subprocess.Popen("tensorboard --logdir ./a2c_poppy_tensorboard/", shell=True)
model.learn(total_timesteps=50000, tb_log_name="Run", callback=callbacks)
os.makedirs("./a2c_poppy_tensorboard/exports", exist_ok=True)
model.save("./a2c_poppy_tensorboard/exports/a2c_poppy")
