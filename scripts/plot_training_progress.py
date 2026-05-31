import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create training progress plots from Stable-Baselines3 monitor CSV files."
    )

    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing training run folders.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory where plots should be saved.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling window size for smoothing rewards.",
    )

    return parser.parse_args()


def find_monitor_files(runs_dir: Path):
    """
    Finds Stable-Baselines3 monitor files inside runs/* directories.

    Examples:
    runs/ppo_100k_obs_v2_seed_42/training.monitor.csv
    runs/ppo_100k_obs_v2_seed_42/monitor.monitor.csv
    """
    patterns = [
        "*.monitor.csv",
        "*monitor*.csv",
    ]

    files = []

    for pattern in patterns:
        files.extend(runs_dir.glob(f"*/{pattern}"))

    unique_files = []
    seen = set()

    for file_path in files:
        if file_path not in seen:
            unique_files.append(file_path)
            seen.add(file_path)

    return unique_files


def read_monitor_csv(file_path: Path):
    """
    Reads Stable-Baselines3 Monitor CSV.

    Monitor files usually start with metadata line:
    #{"t_start": ...}

    Then they contain columns:
    r,l,t

    r - episode reward
    l - episode length
    t - elapsed time
    """
    return pd.read_csv(file_path, comment="#")


def prepare_training_curve(df: pd.DataFrame):
    required_columns = {"r", "l"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Monitor CSV must contain columns {required_columns}, "
            f"but got {set(df.columns)}"
        )

    df = df.copy()

    # Timesteps are calculated as cumulative episode lengths.
    df["timesteps"] = df["l"].cumsum()

    return df


def plot_single_run(monitor_file: Path, output_dir: Path, window: int):
    run_name = monitor_file.parent.name

    df = read_monitor_csv(monitor_file)
    df = prepare_training_curve(df)

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["timesteps"],
        df["r"],
        alpha=0.3,
        label="Raw episode reward",
    )

    if len(df) >= window:
        rolling_reward = df["r"].rolling(window=window).mean()

        plt.plot(
            df["timesteps"],
            rolling_reward,
            linewidth=2,
            label=f"Rolling mean ({window})",
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.title(f"Training progress: {run_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"{run_name}_training_progress.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved plot: {output_path}")


def main():
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        print(f"Runs directory does not exist: {runs_dir}")
        sys.exit(1)

    monitor_files = find_monitor_files(runs_dir)

    if not monitor_files:
        print(f"No monitor CSV files found in: {runs_dir}")
        print("Expected files like:")
        print("  runs/ppo_100k_obs_v2_seed_42/training.monitor.csv")
        print("  runs/ppo_100k_obs_v2_seed_42/monitor.monitor.csv")
        sys.exit(1)

    print("Found monitor files:")
    for monitor_file in monitor_files:
        print(f"- {monitor_file}")

    for monitor_file in monitor_files:
        plot_single_run(
            monitor_file=monitor_file,
            output_dir=output_dir,
            window=args.window,
        )


if __name__ == "__main__":
    main()