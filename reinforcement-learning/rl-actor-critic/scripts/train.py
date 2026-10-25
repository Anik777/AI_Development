#!/usr/bin/env python3
"""Command-line entry point for LunarLander actor-critic training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_actor_critic import TrainingConfig, train  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an advantage actor-critic agent on LunarLander-v3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "lunar_lander.json",
        help="JSON file supplying base hyperparameters",
    )
    parser.add_argument("--run-name", help="optional suffix for the run directory")
    parser.add_argument("--run-dir", type=Path, help="exact output directory")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float, dest="gae_lambda")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate")
    parser.add_argument("--hidden-size", type=int, dest="hidden_size")
    parser.add_argument("--value-loss-coef", type=float, dest="value_loss_coef")
    parser.add_argument("--entropy-coef", type=float, dest="entropy_coef")
    parser.add_argument("--max-grad-norm", type=float, dest="max_grad_norm")
    parser.add_argument(
        "--max-steps-per-episode", type=int, dest="max_steps_per_episode"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluation-seed", type=int, dest="evaluation_seed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--rolling-window", type=int, dest="rolling_window")
    parser.add_argument("--log-interval", type=int, dest="log_interval")
    parser.add_argument(
        "--checkpoint-interval", type=int, dest="checkpoint_interval"
    )
    parser.add_argument(
        "--record-episodes",
        type=int,
        nargs="+",
        dest="record_episodes",
        help="evaluation/video milestones, where 0 is the untrained policy",
    )
    parser.add_argument("--solved-score", type=float, dest="solved_score")

    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument(
        "--video", action="store_true", dest="record_video", default=None
    )
    video_group.add_argument(
        "--no-video", action="store_false", dest="record_video"
    )
    solved_group = parser.add_mutually_exclusive_group()
    solved_group.add_argument(
        "--stop-on-solved", action="store_true", dest="stop_on_solved", default=None
    )
    solved_group.add_argument(
        "--no-stop-on-solved", action="store_false", dest="stop_on_solved"
    )
    deterministic_group = parser.add_mutually_exclusive_group()
    deterministic_group.add_argument(
        "--deterministic-torch",
        action="store_true",
        dest="deterministic_torch",
        default=None,
    )
    deterministic_group.add_argument(
        "--non-deterministic-torch",
        action="store_false",
        dest="deterministic_torch",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig.from_json(args.config)
    override_names = (
        "episodes",
        "gamma",
        "gae_lambda",
        "learning_rate",
        "hidden_size",
        "value_loss_coef",
        "entropy_coef",
        "max_grad_norm",
        "max_steps_per_episode",
        "seed",
        "evaluation_seed",
        "device",
        "rolling_window",
        "log_interval",
        "checkpoint_interval",
        "record_episodes",
        "record_video",
        "deterministic_torch",
        "solved_score",
        "stop_on_solved",
    )
    overrides = {name: getattr(args, name) for name in override_names}
    config = config.with_overrides(**overrides)
    result = train(
        config,
        project_root=PROJECT_ROOT,
        run_directory=args.run_dir,
        run_name=args.run_name,
    )
    print()
    print("Run complete")
    print("------------")
    print(f"Directory: {result.run_directory}")
    print(f"Episodes: {result.completed_episodes}")
    print(f"Best reward: {result.best_reward:.3f}")
    print(f"Final rolling reward: {result.final_rolling_reward:.3f}")
    print(f"Checkpoint: {result.final_checkpoint}")


if __name__ == "__main__":
    main()

