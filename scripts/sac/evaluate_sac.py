import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from stable_baselines3 import SAC

from environment.swarmball_env import SwarmBall


def evaluate(args):
    """Evaluate a trained SAC model."""

    env = SwarmBall(
        render_mode=None,
        acc_factor=0.12,
        v_max=8,
        goal_target=args.goal_target,
        enemy_acceleration=args.enemy_acceleration,
        enemy_max_speed=args.enemy_max_speed,
    )

    model = SAC.load(args.model_path, env=env, device=args.device)

    rewards = []
    episode_lengths = []
    success_count = 0
    enemy_caught_count = 0
    time_limit_count = 0

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        last_info = {}

        while not terminated and not truncated and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1
            last_info = info

        rewards.append(total_reward)
        episode_lengths.append(steps)

        if last_info.get("is_success", False):
            success_count += 1

        if last_info.get("enemy_caught", False):
            enemy_caught_count += 1

        if not last_info.get("is_success", False) and not last_info.get("enemy_caught", False):
            time_limit_count += 1

        print(
            f"Episode {episode + 1}/{args.episodes}: "
            f"reward={total_reward:.2f}, "
            f"steps={steps}, "
            f"success={last_info.get('is_success', False)}, "
            f"enemy_caught={last_info.get('enemy_caught', False)}"
        )

    env.close()

    rewards_np = np.array(rewards, dtype=np.float32)
    lengths_np = np.array(episode_lengths, dtype=np.float32)

    print()
    print("SAC Evaluation summary")
    print(f"Mean reward: {rewards_np.mean():.2f}")
    print(f"Median reward: {np.median(rewards_np):.2f}")
    print(f"Min reward: {rewards_np.min():.2f}")
    print(f"Max reward: {rewards_np.max():.2f}")
    print(f"Std reward: {rewards_np.std():.2f}")
    print(f"Mean episode length: {lengths_np.mean():.2f}")
    print(f"Success count: {success_count}/{args.episodes}")
    print(f"Enemy caught count: {enemy_caught_count}/{args.episodes}")
    print(f"Time limit count: {time_limit_count}/{args.episodes}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC model.")

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--goal-target", type=float, default=500.0)
    parser.add_argument("--enemy-acceleration", type=float, default=0.003)
    parser.add_argument("--enemy-max-speed", type=float, default=3.0)

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())