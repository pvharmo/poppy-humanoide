from stable_baselines3 import A2C

from environment import RobotEnv

env = RobotEnv()
print("Environment initialized")

model = A2C("MlpPolicy", env, verbose=1)
print("Model initialized")
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")
