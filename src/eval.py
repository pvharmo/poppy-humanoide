from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from mujoco_env import PoppyEnv
from paths import exports_path, scene_path, videos_path

def make_env():
    return PoppyEnv(scene_path, render_mode="rgb_array")


env = DummyVecEnv([make_env])

env = VecNormalize.load(exports_path + "/vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

env.envs[0].metadata['render_fps'] = 120

env = VecVideoRecorder(
    env,
    video_folder=videos_path,
    record_video_trigger=lambda step: step == 0,
    video_length=4000,
    name_prefix="ppo_h_stand_eval"
)


model = PPO.load(exports_path + "/ppo_poppy_final", env=env, device="cpu")

obs = env.reset()
rewards = 0

for i in range(1000):  # 1000 timesteps
    action, _ = model.predict(obs, deterministic=True) # pyright: ignore[reportArgumentType]
    obs, reward, dones, _ = env.step(action)
    rewards += reward

    # if dones.any():
    #     break

env.close()
print(f"Video saved in {videos_path}")
