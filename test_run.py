"""
run_trained.py - Uruchamia wytrenowany model i wizualizuje go

Lokalnie (z oknem pygame):
    python3 run_trained.py

W Dockerze (headless, tylko logi):
    docker compose run --rm gpu python3 run_trained.py --no-render
"""
import os
import time
import argparse
import numpy as np
from stable_baselines3 import PPO

try:
    from environment.swarmball_env import SwarmBall
except ImportError:
    from swarmball_env import SwarmBall


def run(args):
    render_mode = None if args.no_render else "human"

    if not args.no_render:
        # Tryb graficzny - potrzebny display
        pass
    else:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    env = SwarmBall(
        number_of_clusters=3,
        number_of_bots_per_cluster=10,
        goal_target=1500.0,
        enemy_max_speed=1.5,
        enemy_acceleration=0.002,
        render_mode=render_mode,
    )

    model_path = args.model_path
    if not os.path.exists(f"{model_path}.zip"):
        # Spróbuj też best model
        best_path = "models/best/best_model"
        if os.path.exists(f"{best_path}.zip"):
            model_path = best_path
            print(f"Używam najlepszego modelu: {best_path}.zip")
        else:
            print(f"BŁĄD: Nie znaleziono modelu pod {model_path}.zip")
            print("Uruchom najpierw: python3 train_ppo.py")
            return

    print(f"Wczytuję model: {model_path}.zip")
    model = PPO.load(model_path, device="cpu")

    total_rewards = []
    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0.0
        step = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if not args.no_render:
                env.render()
                time.sleep(0.002)  

            if terminated or truncated:
                break

        total_rewards.append(total_reward)
        result = "SUKCES (enemy dogonił)" if terminated else "LIMIT CZASU"
        print(
            f"Epizod {episode+1:>3}: "
            f"nagroda={total_reward:>8.1f}, "
            f"kroki={step:>4}, "
            f"wynik={result}"
        )

    print(f"\nŚrednia nagroda: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmBall - uruchomienie modelu")
    parser.add_argument(
        "--model-path", type=str, default="models/ppo_swarmball",
        help="Ścieżka do modelu (bez .zip)"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Liczba epizodów do uruchomienia (default: 10)"
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Wyłącz renderowanie (headless)"
    )
    args = parser.parse_args()
    run(args)