"""On-policy advantage actor-critic training with generalized advantage estimation."""

from __future__ import annotations

import random
import shlex
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .config import TrainingConfig
from .evaluation import EvaluationResult, evaluate_policy
from .model import ActorCritic
from .run_io import (
    IterationLogWriter,
    MetricWriter,
    append_jsonl,
    collect_metadata,
    configure_logger,
    create_run_directory,
    prepare_run_directory,
    utc_now,
    write_evaluation_step_plots,
    write_json,
    write_reward_svg,
    write_training_plots,
)


@dataclass(slots=True)
class TrainingResult:
    run_directory: Path
    final_checkpoint: Path
    completed_episodes: int
    best_reward: float
    final_rolling_reward: float
    device: str


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int, deterministic_torch: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)


def compute_gae(
    rewards: list[float],
    values: list[torch.Tensor],
    bootstrap_masks: list[float],
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantages and value targets in reverse time."""

    advantages: list[torch.Tensor] = []
    gae = torch.zeros_like(last_value)
    next_value = last_value.detach()

    for reward, value, mask in zip(
        reversed(rewards), reversed(values), reversed(bootstrap_masks), strict=True
    ):
        reward_tensor = value.new_tensor(reward)
        mask_tensor = value.new_tensor(mask)
        delta = reward_tensor + gamma * next_value * mask_tensor - value.detach()
        gae = delta + gamma * gae_lambda * mask_tensor * gae
        advantages.append(gae)
        next_value = value.detach()

    advantages.reverse()
    advantage_tensor = torch.stack(advantages)
    value_tensor = torch.stack(values)
    returns = advantage_tensor + value_tensor.detach()
    return advantage_tensor, returns


def save_checkpoint(
    path: Path,
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    episode: int,
    observation_size: int,
    action_size: int,
    metrics: dict[str, Any] | None,
) -> None:
    torch.save(
        {
            "episode": episode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "observation_size": observation_size,
            "action_size": action_size,
            "metrics": metrics,
        },
        path,
    )


def train(
    config: TrainingConfig,
    *,
    project_root: str | Path | None = None,
    run_directory: str | Path | None = None,
    run_name: str | None = None,
) -> TrainingResult:
    """Train the agent and persist every run artifact."""

    config.validate()
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    run_dir = (
        Path(run_directory).resolve()
        if run_directory is not None
        else create_run_directory(root, run_name)
    )
    prepare_run_directory(run_dir)
    logger = configure_logger(run_dir)
    device = select_device(config.device)
    seed_everything(config.seed, config.deterministic_torch)

    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "metadata.json", collect_metadata(device))
    (run_dir / "command.txt").write_text(
        shlex.join(sys.argv) + "\n", encoding="utf-8"
    )
    latest_file = root / "runs" / "latest_run.txt"
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(str(run_dir) + "\n", encoding="utf-8")

    env = gym.make(config.env_id)
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("This example requires a Box observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("This example requires a Discrete action space")
    observation_size = int(np.prod(env.observation_space.shape))
    action_size = int(env.action_space.n)

    model = ActorCritic(observation_size, action_size, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    events_path = run_dir / "events.jsonl"
    history: list[dict[str, Any]] = []
    rolling_rewards: deque[float] = deque(maxlen=config.rolling_window)
    best_reward = float("-inf")
    solved_at: int | None = None
    started = time.perf_counter()
    completed_episodes = 0

    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)
    logger.info("Environment: %s", config.env_id)
    logger.info(
        "Episodes=%d gamma=%.4f gae_lambda=%.4f learning_rate=%g",
        config.episodes,
        config.gamma,
        config.gae_lambda,
        config.learning_rate,
    )
    append_jsonl(events_path, {"type": "run_started", "time": utc_now()})

    record_set = set(config.record_episodes)
    if 0 in record_set:
        initial_eval = evaluate_policy(
            model,
            env_id=config.env_id,
            device=device,
            seed=config.evaluation_seed,
            max_steps=config.max_steps_per_episode,
            label="episode_0000",
            video_directory=run_dir / "videos" if config.record_video else None,
        )
        append_jsonl(
            events_path,
            {"type": "evaluation", "time": utc_now(), **initial_eval.to_dict()},
        )
        write_evaluation_step_plots(
            label=initial_eval.label,
            step_rewards=initial_eval.step_rewards,
            plot_directory=run_dir / "plots" / "evaluations",
        )
        logger.info("Evaluation episode 0 reward: %.3f", initial_eval.reward)

    last_metrics: dict[str, Any] | None = None

    try:
        with (
            MetricWriter(run_dir / "metrics.csv") as metric_writer,
            IterationLogWriter(run_dir / "iterations.log") as iteration_log_writer,
        ):
            for episode in range(1, config.episodes + 1):
                observation, _ = env.reset(seed=config.seed + episode - 1)
                log_probabilities: list[torch.Tensor] = []
                values: list[torch.Tensor] = []
                entropies: list[torch.Tensor] = []
                rewards: list[float] = []
                bootstrap_masks: list[float] = []
                episode_reward = 0.0
                terminated = False
                truncated = False

                for step in range(1, config.max_steps_per_episode + 1):
                    state = torch.as_tensor(
                        np.asarray(observation), dtype=torch.float32, device=device
                    )
                    action, log_probability, entropy, value = model.choose_action(state)
                    next_observation, reward, terminated, env_truncated, _ = env.step(
                        int(action.item())
                    )
                    forced_truncation = (
                        step >= config.max_steps_per_episode and not terminated
                    )
                    truncated = bool(env_truncated or forced_truncation)

                    log_probabilities.append(log_probability)
                    values.append(value)
                    entropies.append(entropy)
                    rewards.append(float(reward))
                    bootstrap_masks.append(0.0 if terminated else 1.0)
                    episode_reward += float(reward)
                    observation = next_observation

                    if terminated or truncated:
                        break

                with torch.no_grad():
                    if terminated:
                        last_value = torch.zeros((), dtype=torch.float32, device=device)
                    else:
                        final_state = torch.as_tensor(
                            np.asarray(observation), dtype=torch.float32, device=device
                        )
                        _, last_value = model(final_state)

                advantages, returns = compute_gae(
                    rewards,
                    values,
                    bootstrap_masks,
                    last_value,
                    config.gamma,
                    config.gae_lambda,
                )
                if advantages.numel() > 1:
                    normalized_advantages = (
                        advantages - advantages.mean()
                    ) / (advantages.std(unbiased=False) + 1e-8)
                else:
                    normalized_advantages = advantages

                log_probability_tensor = torch.stack(log_probabilities)
                entropy_tensor = torch.stack(entropies)
                value_tensor = torch.stack(values)
                actor_loss = -(
                    log_probability_tensor * normalized_advantages.detach()
                ).mean()
                critic_loss = 0.5 * (returns.detach() - value_tensor).pow(2).mean()
                mean_entropy = entropy_tensor.mean()
                total_loss = (
                    actor_loss
                    + config.value_loss_coef * critic_loss
                    - config.entropy_coef * mean_entropy
                )

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()

                completed_episodes = episode
                best_reward = max(best_reward, episode_reward)
                rolling_rewards.append(episode_reward)
                rolling_reward = float(np.mean(rolling_rewards))
                elapsed = time.perf_counter() - started
                last_metrics = {
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "episode_length": len(rewards),
                    "rolling_reward": rolling_reward,
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "critic_loss": float(critic_loss.detach().cpu()),
                    "entropy": float(mean_entropy.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                    "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": elapsed,
                    "terminated": terminated,
                    "truncated": truncated,
                    "device": str(device),
                }
                history.append(last_metrics)
                metric_writer.write(last_metrics)
                iteration_log_writer.write(last_metrics)
                append_jsonl(
                    events_path,
                    {"type": "training_iteration", "time": utc_now(), **last_metrics},
                )

                if episode == 1 or episode % config.log_interval == 0:
                    logger.info(
                        "episode=%d reward=%.3f rolling=%.3f steps=%d "
                        "actor_loss=%.5f critic_loss=%.5f entropy=%.5f",
                        episode,
                        episode_reward,
                        rolling_reward,
                        len(rewards),
                        last_metrics["actor_loss"],
                        last_metrics["critic_loss"],
                        last_metrics["entropy"],
                    )

                should_checkpoint = (
                    episode == 1
                    or episode in record_set
                    or episode % config.checkpoint_interval == 0
                )
                if should_checkpoint:
                    checkpoint = run_dir / "checkpoints" / f"episode_{episode:04d}.pt"
                    save_checkpoint(
                        checkpoint,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        episode=episode,
                        observation_size=observation_size,
                        action_size=action_size,
                        metrics=last_metrics,
                    )
                    logger.info("Saved checkpoint: %s", checkpoint.name)

                if episode in record_set:
                    evaluation = evaluate_policy(
                        model,
                        env_id=config.env_id,
                        device=device,
                        seed=config.evaluation_seed,
                        max_steps=config.max_steps_per_episode,
                        label=f"episode_{episode:04d}",
                        video_directory=(
                            run_dir / "videos" if config.record_video else None
                        ),
                    )
                    append_jsonl(
                        events_path,
                        {
                            "type": "evaluation",
                            "time": utc_now(),
                            **evaluation.to_dict(),
                        },
                    )
                    write_evaluation_step_plots(
                        label=evaluation.label,
                        step_rewards=evaluation.step_rewards,
                        plot_directory=run_dir / "plots" / "evaluations",
                    )
                    logger.info(
                        "Evaluation episode %d reward: %.3f", episode, evaluation.reward
                    )

                if rolling_reward >= config.solved_score and solved_at is None:
                    solved_at = episode
                    logger.info(
                        "Solved threshold %.1f reached at episode %d",
                        config.solved_score,
                        episode,
                    )
                    if config.stop_on_solved:
                        break
    except Exception as error:
        logger.exception("Training failed")
        append_jsonl(
            events_path,
            {
                "type": "run_failed",
                "time": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        env.close()

    final_checkpoint = run_dir / "checkpoints" / "final.pt"
    save_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        episode=completed_episodes,
        observation_size=observation_size,
        action_size=action_size,
        metrics=last_metrics,
    )
    final_evaluation: EvaluationResult = evaluate_policy(
        model,
        env_id=config.env_id,
        device=device,
        seed=config.evaluation_seed,
        max_steps=config.max_steps_per_episode,
        label="final",
        video_directory=run_dir / "videos" if config.record_video else None,
    )
    append_jsonl(
        events_path,
        {"type": "evaluation", "time": utc_now(), **final_evaluation.to_dict()},
    )
    write_evaluation_step_plots(
        label=final_evaluation.label,
        step_rewards=final_evaluation.step_rewards,
        plot_directory=run_dir / "plots" / "evaluations",
    )
    elapsed = time.perf_counter() - started
    final_rolling = float(np.mean(rolling_rewards)) if rolling_rewards else float("nan")
    summary = {
        "completed_at_utc": utc_now(),
        "completed_episodes": completed_episodes,
        "best_training_reward": best_reward,
        "final_rolling_reward": final_rolling,
        "final_evaluation": final_evaluation.to_dict(),
        "solved_at_episode": solved_at,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "final_checkpoint": str(final_checkpoint),
    }
    write_json(run_dir / "summary.json", summary)
    write_reward_svg(history, run_dir / "reward_curve.svg")
    write_training_plots(history, run_dir / "plots")
    append_jsonl(events_path, {"type": "run_completed", "time": utc_now(), **summary})
    logger.info(
        "Training complete: episodes=%d best_reward=%.3f final_eval=%.3f elapsed=%.1fs",
        completed_episodes,
        best_reward,
        final_evaluation.reward,
        elapsed,
    )

    return TrainingResult(
        run_directory=run_dir,
        final_checkpoint=final_checkpoint,
        completed_episodes=completed_episodes,
        best_reward=best_reward,
        final_rolling_reward=final_rolling,
        device=str(device),
    )

