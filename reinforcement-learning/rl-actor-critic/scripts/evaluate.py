#!/usr/bin/env python3
"""Evaluate a saved actor-critic checkpoint and optionally record it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_actor_critic.evaluation import evaluate_policy  # noqa: E402
from rl_actor_critic.model import ActorCritic  # noqa: E402
from rl_actor_critic.trainer import select_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved LunarLander actor-critic checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--label", default="manual")
    args = parser.parse_args()

    device = select_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ActorCritic(
        payload["observation_size"],
        payload["action_size"],
        payload["config"]["hidden_size"],
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    result = evaluate_policy(
        model,
        env_id=payload["config"]["env_id"],
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        label=args.label,
        video_directory=args.video_dir,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

