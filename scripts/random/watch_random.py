import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from environment.swarmball_env import SwarmBall


def watch(args):
    """Run visual simulation using random actions."""

    env = SwarmBall(
        render_mode="human",
        acc_factor=0.12,
        v_max=8,
        goal_target=args.goal_target,
        enemy_acceleration=args.enemy_acceleration,
        enemy_max_speed=args.enemy_max_speed,
    )

    episode = 0

    while True:
        obs, info = env.reset(seed=args.seed + episode)

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        print(f"Starting random episode {episode + 1}")

        while not terminated and not truncated and steps < args.max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1

            env.render()

            if args.sleep > 0:
                time.sleep(args.sleep)

        print(
            f"Random episode {episode + 1} finished: "
            f"reward={total_reward:.2f}, "
            f"steps={steps}, "
            f"terminated={terminated}, "
            f"truncated={truncated}, "
            f"success={info.get('is_success', False)}, "
            f"enemy_caught={info.get('enemy_caught', False)}, "
            f"goal_progress_pct={info.get('goal_progress_pct', None)}"
        )

        episode += 1

        if not args.reset_on_done:
            break

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Watch random baseline.")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--reset-on-done", action="store_true")

    parser.add_argument("--goal-target", type=float, default=500.0)
    parser.add_argument("--enemy-acceleration", type=float, default=0.003)
    parser.add_argument("--enemy-max-speed", type=float, default=3.0)

    return parser.parse_args()


if __name__ == "__main__":
    watch(parse_args())