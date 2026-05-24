import gymnasium as gym
from stable_baselines3 import PPO
from environment.swarmball_env import SwarmBall

# Inicjalizacja środowiska
env = SwarmBall(number_of_clusters=3)

# TO JEST KLUCZ: Używamy tego samego wrappera, co w treningu!
env = gym.wrappers.FlattenObservation(env)

# Wczytaj model
model = PPO.load("ppo_swarmball_model")

# Reset środowiska (zwróci już spłaszczoną obserwację dzięki wrapperowi)
obs, _ = env.reset()

# Symuluj 500 kroków
for i in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Drukujemy, żeby widzieć, że agent faktycznie podejmuje decyzje
    print(f"Krok {i}, Nagroda: {reward:.2f}")
    
    if terminated or truncated:
        break