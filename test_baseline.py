import gymnasium as gym
from stable_baselines3 import PPO
from environment.swarmball_env import SwarmBall 

env = SwarmBall(number_of_clusters=3)

env = gym.wrappers.FlattenObservation(env)

print("Environment loaded and flattened!")

try:
    model = PPO("MlpPolicy", env, verbose=1)
    print("PPO model initialized!")
    
    model.learn(total_timesteps=50000)
    print("Training test completed successfully!")
    
    model.save("ppo_swarmball_model")
    
except Exception as e:
    print(f"An error occurred during training: {e}")