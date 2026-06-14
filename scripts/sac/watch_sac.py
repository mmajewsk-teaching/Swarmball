import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from stable_baselines3 import SAC

from environment.swarmball_env import SwarmBall


def watch(args):
    """Run visual simulation using a trained SAC model."""

    env = SwarmBall(
        render_mode="human",
        acc_factor=0.12,
        v_max=8,
        goal_target=args.goal_target,
        enemy_acceleration=args.enemy_acceleration,
        enemy_max_speed=args.enemy_max_speed,
    )

    model = SAC.load(args.model_path, env=env, device=args.device)

    episode = 0

    while True:
        obs, info = env.reset(seed=args.seed + episode)

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        print(f"Starting episode {episode + 1}")

        while not terminated and not truncated and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1

            env.render()

            if args.sleep > 0:
                time.sleep(args.sleep)

        print(
            f"Episode {episode + 1} finished: "
            f"reward={total_reward:.2f}, "
            f"steps={steps}, "
            f"terminated={terminated}, "
            f"truncated={truncated}, "
            f"success={info.get('is_success', False)}, "
            f"enemy_caught={info.get('enemy_caught', False)}, "
            f"info={info}"
        )

        episode += 1

        if not args.reset_on_done:
            break

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained SAC model.")

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--reset-on-done", action="store_true")

    parser.add_argument("--goal-target", type=float, default=500.0)
    parser.add_argument("--enemy-acceleration", type=float, default=0.003)
    parser.add_argument("--enemy-max-speed", type=float, default=3.0)

    return parser.parse_args()


if __name__ == "__main__":
    watch(parse_args())