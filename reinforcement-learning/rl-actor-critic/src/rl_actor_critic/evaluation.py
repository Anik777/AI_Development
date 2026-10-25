"""Deterministic policy evaluation and optional MP4 recording."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import warnings

import gymnasium as gym
import numpy as np
import torch

from .model import ActorCritic


@dataclass(slots=True)
class EvaluationResult:
    label: str
    reward: float
    steps: int
    terminated: bool
    truncated: bool
    video_file: str | None
    step_rewards: list[float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_policy(
    model: ActorCritic,
    *,
    env_id: str,
    device: torch.device,
    seed: int,
    max_steps: int,
    label: str,
    video_directory: Path | None = None,
) -> EvaluationResult:
    """Run one greedy episode with a fixed seed and no learning."""

    render_mode = "rgb_array" if video_directory is not None else None
    env = gym.make(env_id, render_mode=render_mode)
    video_file: str | None = None
    before: set[Path] = set()

    if video_directory is not None:
        video_directory.mkdir(parents=True, exist_ok=True)
        before = set(video_directory.glob("*.mp4"))
        # RecordVideo warns whenever its folder already exists, even though our
        # label-specific prefixes do not overwrite earlier checkpoint videos.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Overwriting existing videos at.*"
            )
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_directory),
                episode_trigger=lambda _: True,
                name_prefix=f"evaluation_{label}",
                disable_logger=True,
            )

    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    step_rewards: list[float] = []
    steps = 0
    terminated = False
    truncated = False
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            while not (terminated or truncated) and steps < max_steps:
                state = torch.as_tensor(
                    np.asarray(observation), dtype=torch.float32, device=device
                )
                action, _, _, _ = model.choose_action(state, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(
                    int(action.item())
                )
                total_reward += float(reward)
                step_rewards.append(float(reward))
                steps += 1
            if steps >= max_steps and not terminated:
                truncated = True
    finally:
        env.close()
        model.train(was_training)

    if video_directory is not None:
        created = sorted(set(video_directory.glob("*.mp4")) - before)
        if created:
            video_file = str(created[-1])

    return EvaluationResult(
        label=label,
        reward=total_reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        video_file=video_file,
        step_rewards=step_rewards,
    )

