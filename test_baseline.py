import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

# Headless pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

try:
    from environment.swarmball_env import SwarmBall
except ImportError:
    from swarmball_env import SwarmBall


# -------------------------------------------------------------------------
# Callback z rozszerzonymi metrykami
# -------------------------------------------------------------------------
class TrainingMetricsCallback(BaseCallback):
    """Loguje metryki + postęp % do mety."""

    def __init__(self, print_freq=10000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= self.print_freq:
            self._last_print = self.num_timesteps
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                ep_lens    = [ep["l"] for ep in self.model.ep_info_buffer]
                print(
                    f"[{self.num_timesteps:>8} steps] "
                    f"mean_reward={np.mean(ep_rewards):>8.1f}  "
                    f"mean_len={np.mean(ep_lens):>6.0f}  "
                    f"best_reward={np.max(ep_rewards):>8.1f}"
                )
        return True


# -------------------------------------------------------------------------
# Fabryka środowiska
# -------------------------------------------------------------------------
def make_env(rank: int = 0, seed: int = 0, goal_target: float = 300.0):
    def _init():
        env = SwarmBall(
            number_of_clusters=3,
            number_of_bots_per_cluster=10,
            acc_factor=0.12,
            v_max=8,
            goal_target=goal_target,
            enemy_acceleration=0.003,
            enemy_max_speed=3.0,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# -------------------------------------------------------------------------
# Trening
# -------------------------------------------------------------------------
def train(args):
    print("=" * 80)
    print("SwarmBall PPO Training")
    print("=" * 80)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    n_envs = args.n_envs
    goal_target = args.goal_target
    print(f"Tworzę {n_envs} równoległych środowisk (cel: {goal_target})...")

    if n_envs > 1:
        vec_env = SubprocVecEnv([make_env(i, goal_target=goal_target) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(0, goal_target=goal_target)])

    eval_env = DummyVecEnv([make_env(99, seed=42, goal_target=goal_target)])

    # ---- Hiperparametry ----
    n_steps = 2048
    total_steps_per_update = n_steps * n_envs
    batch_size = 256
    for bs in [512, 256, 128, 64, 32]:
        if total_steps_per_update % bs == 0:
            batch_size = bs
            break

    print(f"n_steps={n_steps}, n_envs={n_envs}, batch_size={batch_size}")

    model_path = args.model_path
    model = None

    if os.path.exists(f"{model_path}.zip") and not args.force_new:
        print(f"\nWczytuję istniejący model z {model_path}.zip ...")
        try:
            model = PPO.load(model_path, env=vec_env, device=args.device,
                             learning_rate=args.lr)
            print("Model wczytany, kontynuuję trening.")
        except (ValueError, Exception) as exc:
            print(f"Niekompatybilny model ({exc}), tworzę nowy.")
            model = None

    if model is None:
        print("\nTworzę nowy model PPO...")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            device=args.device,
            learning_rate=args.lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.995,        # długi horyzont - ważny dla zadania ciągłego
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,     # mała entropia - model już coś umie, nie eksploruj za dużo
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[256, 256, 128],
            ),
            tensorboard_log="logs/tensorboard",
        )

    print("\nArchitektura:")
    print(model.policy)

    # ---- Callbacki ----
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.timesteps // 20, 10000),
        save_path="models/checkpoints/",
        name_prefix="swarmball_ppo",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/best/",
        log_path="logs/eval/",
        eval_freq=max(args.timesteps // 10, 5000),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    metrics_cb = TrainingMetricsCallback(
        print_freq=max(args.timesteps // 100, 2000)
    )

    # ---- Trening ----
    print(f"\nStart: {args.timesteps:,} timesteps na '{args.device}'")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb, metrics_cb],
            progress_bar=False,
            reset_num_timesteps=not os.path.exists(f"{model_path}.zip"),
        )
    except KeyboardInterrupt:
        print("\n\nTrening przerwany.")

    model.save(model_path)
    print(f"\nModel zapisany: {model_path}.zip")

    print("\n--- Ewaluacja końcowa (10 epizodów) ---")
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Średnia nagroda: {mean_r:.2f} ± {std_r:.2f}")

    vec_env.close()
    eval_env.close()


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmBall PPO Training")
    parser.add_argument("--timesteps",  type=int,   default=2_000_000,
                        help="Liczba kroków (default: 2_000_000 - potrzeba min. 2M!)")
    parser.add_argument("--n-envs",     type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--device",     type=str,   default="auto")
    parser.add_argument("--model-path", type=str,   default="models/ppo_swarmball")
    parser.add_argument("--goal-target",type=float, default=300.0,
                        help="Pozycja X mety (sprawdź w symulacji ile wynosi!)")
    parser.add_argument("--force-new",  action="store_true",
                        help="Zacznij od nowa mimo istniejącego modelu")
    args = parser.parse_args()
    train(args)