import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import the custom Gomoku environment
from gomoku_env import GomokuEnv

# Create the environment
env = GomokuEnv()

# Check if the environment follows the Gym API
check_env(env)

# Create the PPO model
model = PPO('CnnPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=1000000)

# Save the model
model.save("ppo_gomoku_8x8_1")

# Load the model
model = PPO.load("ppo_gomoku_8x8_1")

# Test the trained model
obs = env.reset()
for _ in range(500):
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break
