import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.swarmball_env import SwarmBall


def make_env(args, seed: int, monitor_dir: Path | None = None):
    """Create one monitored SwarmBall environment."""

    def _init():
        env = SwarmBall(
            render_mode=None,
            acc_factor=0.12,
            v_max=8,
            goal_target=args.goal_target,
            enemy_acceleration=args.enemy_acceleration,
            enemy_max_speed=args.enemy_max_speed,
        )

        env.reset(seed=seed)

        if monitor_dir is not None:
            monitor_dir.mkdir(parents=True, exist_ok=True)
            env = Monitor(env, str(monitor_dir / "env.monitor.csv"))

        return env

    return _init


def create_model(args, env, tensorboard_dir: Path) -> SAC:
    """Create a new SAC model."""

    return SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device=args.device,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        tensorboard_log=str(tensorboard_dir),
        seed=args.seed,
    )


def save_config(args, run_dir: Path):
    """Save training configuration to JSON."""

    config_path = run_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=4)


def train(args):
    """Run SAC training."""

    run_dir = Path("runs") / args.run_name
    model_path = run_dir / "model.zip"
    best_model_dir = run_dir / "best_model"
    checkpoints_dir = run_dir / "checkpoints"
    monitor_dir = run_dir / "monitor"
    eval_monitor_dir = run_dir / "eval_monitor"
    tensorboard_dir = run_dir / "tensorboard"

    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    eval_monitor_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    save_config(args, run_dir)

    print("SwarmBall SAC Training")
    print(f"Run directory: {run_dir}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Goal target: {args.goal_target}")
    print(f"Enemy acceleration: {args.enemy_acceleration}")
    print(f"Enemy max speed: {args.enemy_max_speed}")

    env = DummyVecEnv([
        make_env(
            args=args,
            seed=args.seed,
            monitor_dir=monitor_dir,
        )
    ])

    eval_env = DummyVecEnv([
        make_env(
            args=args,
            seed=args.seed + 1000,
            monitor_dir=eval_monitor_dir,
        )
    ])

    if args.continue_training and model_path.exists():
        print(f"Loading existing SAC model from {model_path}")
        model = SAC.load(str(model_path), env=env, device=args.device)
    else:
        model = create_model(
            args=args,
            env=env,
            tensorboard_dir=tensorboard_dir,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoints_dir),
        name_prefix="swarmball_sac",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,
    )

    model.save(str(model_path))
    model.save_replay_buffer(str(run_dir / "replay_buffer"))

    env.close()
    eval_env.close()

    print(f"Training finished. Final model saved to: {model_path}")
    print(f"Best model directory: {best_model_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on SwarmBall.")

    parser.add_argument("--run-name", type=str, default="sac_1m_target_500_seed_42")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)

    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", type=str, default="auto")

    parser.add_argument("--goal-target", type=float, default=500.0)
    parser.add_argument("--enemy-acceleration", type=float, default=0.003)
    parser.add_argument("--enemy-max-speed", type=float, default=3.0)

    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)

    parser.add_argument("--continue-training", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())