"""Run directories, structured metrics, metadata, and lightweight plots."""

from __future__ import annotations

import csv
import html
import json
import logging
import math
import platform
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import gymnasium
import numpy
import torch


METRIC_FIELDS = (
    "episode",
    "episode_reward",
    "episode_length",
    "rolling_reward",
    "actor_loss",
    "critic_loss",
    "entropy",
    "total_loss",
    "grad_norm",
    "learning_rate",
    "elapsed_seconds",
    "terminated",
    "truncated",
    "device",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run_directory(project_root: Path, run_name: str | None = None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    suffix = ""
    if run_name:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-.")
        suffix = f"_{safe_name}" if safe_name else ""
    run_dir = project_root / "runs" / f"{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_run_directory(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("checkpoints", "videos", "plots", "plots/evaluations"):
        (run_dir / child).mkdir(exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def configure_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"rl_actor_critic.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


class MetricWriter:
    """Append and immediately flush one CSV record per training iteration."""

    def __init__(self, path: Path):
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=METRIC_FIELDS)
        self._writer.writeheader()
        self._handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow({field: row[field] for field in METRIC_FIELDS})
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "MetricWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class IterationLogWriter:
    """Write one immediately flushed, human-readable line per episode."""

    def __init__(self, path: Path):
        self._handle = path.open("w", encoding="utf-8", buffering=1)

    def write(self, row: dict[str, Any]) -> None:
        self._handle.write(
            f"{utc_now()} | "
            f"episode={int(row['episode']):06d} | "
            f"reward={float(row['episode_reward']):10.3f} | "
            f"rolling={float(row['rolling_reward']):10.3f} | "
            f"steps={int(row['episode_length']):04d} | "
            f"actor_loss={float(row['actor_loss']): .6f} | "
            f"critic_loss={float(row['critic_loss']): .6f} | "
            f"entropy={float(row['entropy']): .6f} | "
            f"total_loss={float(row['total_loss']): .6f} | "
            f"grad_norm={float(row['grad_norm']): .6f} | "
            f"device={row['device']}\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "IterationLogWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def collect_metadata(device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "created_at_utc": utc_now(),
        "command": shlex.join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gymnasium": gymnasium.__version__,
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torch_compiled_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "selected_device": str(device),
    }


def write_reward_svg(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write the legacy top-level reward curve for convenient discovery."""

    data = list(rows)
    if not data:
        return

    write_line_svg(
        x_values=[int(row["episode"]) for row in data],
        series={
            "episode reward": [float(row["episode_reward"]) for row in data],
            "rolling reward": [float(row["rolling_reward"]) for row in data],
        },
        path=path,
        title="LunarLander training rewards",
        x_label="Episode",
        y_label="Reward",
    )


def write_line_svg(
    *,
    x_values: Iterable[float],
    series: dict[str, Iterable[float]],
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Create a dependency-free multi-series SVG line plot."""

    x_data = [float(value) for value in x_values]
    y_series = {
        name: [float(value) for value in values] for name, values in series.items()
    }
    if not x_data or not y_series:
        return
    for name, values in y_series.items():
        if len(values) != len(x_data):
            raise ValueError(f"Series {name!r} does not match x-axis length")

    finite_y = [
        value
        for values in y_series.values()
        for value in values
        if math.isfinite(value)
    ]
    finite_x = [value for value in x_data if math.isfinite(value)]
    if not finite_x or not finite_y:
        return

    width, height = 1_000, 500
    left, right, top, bottom = 85, 35, 40, 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_minimum, x_maximum = min(finite_x), max(finite_x)
    y_minimum, y_maximum = min(finite_y), max(finite_y)
    if x_maximum == x_minimum:
        x_maximum += 1.0
    if y_maximum == y_minimum:
        y_maximum += 1.0
        y_minimum -= 1.0
    padding = (y_maximum - y_minimum) * 0.08
    y_minimum -= padding
    y_maximum += padding

    def point(x_value: float, y_value: float) -> str:
        x = left + (x_value - x_minimum) / (x_maximum - x_minimum) * plot_width
        y = top + (y_maximum - y_value) / (y_maximum - y_minimum) * plot_height
        return f"{x:.2f},{y:.2f}"

    colors = ("#1769aa", "#d1495b", "#2a9d8f", "#7b2cbf", "#f4a261")
    polylines: list[str] = []
    legends: list[str] = []
    for index, (name, values) in enumerate(y_series.items()):
        color = colors[index % len(colors)]
        points = " ".join(
            point(x_value, y_value)
            for x_value, y_value in zip(x_data, values, strict=True)
            if math.isfinite(x_value) and math.isfinite(y_value)
        )
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
            f'points="{points}"/>'
        )
        legend_y = 48 + index * 20
        legends.append(
            f'<line x1="{width-260}" y1="{legend_y}" x2="{width-230}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
            f'<text x="{width-222}" y="{legend_y+4}" font-family="sans-serif" '
            f'font-size="12">{html.escape(name)}</text>'
        )

    zero_line = ""
    if y_minimum <= 0.0 <= y_maximum:
        zero_y = top + (y_maximum / (y_maximum - y_minimum)) * plot_height
        zero_line = (
            f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" '
            f'y2="{zero_y:.2f}" stroke="#999" stroke-dasharray="4 4"/>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>
{zero_line}
<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>
<text x="{width/2}" y="{height-14}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text>
<text x="20" y="{height/2}" text-anchor="middle" transform="rotate(-90 20 {height/2})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>
<text x="{left-8}" y="{top+5}" text-anchor="end" font-family="monospace" font-size="11">{y_maximum:.3f}</text>
<text x="{left-8}" y="{height-bottom}" text-anchor="end" font-family="monospace" font-size="11">{y_minimum:.3f}</text>
<text x="{left}" y="{height-bottom+18}" text-anchor="middle" font-family="monospace" font-size="11">{x_minimum:g}</text>
<text x="{width-right}" y="{height-bottom+18}" text-anchor="middle" font-family="monospace" font-size="11">{x_maximum:g}</text>
{''.join(polylines)}
{''.join(legends)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_training_plots(rows: Iterable[dict[str, Any]], plot_directory: Path) -> None:
    """Create reward and separate loss plots over training episodes."""

    data = list(rows)
    if not data:
        return
    episodes = [int(row["episode"]) for row in data]
    plot_directory.mkdir(parents=True, exist_ok=True)
    write_line_svg(
        x_values=episodes,
        series={
            "episode reward": [float(row["episode_reward"]) for row in data],
            "rolling reward": [float(row["rolling_reward"]) for row in data],
        },
        path=plot_directory / "training_episode_rewards.svg",
        title="Training rewards by episode",
        x_label="Episode",
        y_label="Reward",
    )
    for key, title in (
        ("actor_loss", "Actor loss by episode"),
        ("critic_loss", "Critic loss by episode"),
        ("total_loss", "Total training loss by episode"),
    ):
        write_line_svg(
            x_values=episodes,
            series={key.replace("_", " "): [float(row[key]) for row in data]},
            path=plot_directory / f"training_{key}.svg",
            title=title,
            x_label="Episode",
            y_label="Loss",
        )


def write_evaluation_step_plots(
    *, label: str, step_rewards: Iterable[float], plot_directory: Path
) -> None:
    """Persist immediate and cumulative per-step rewards for one evaluation."""

    rewards = [float(reward) for reward in step_rewards]
    if not rewards:
        return
    steps = list(range(1, len(rewards) + 1))
    cumulative: list[float] = []
    running_total = 0.0
    for reward in rewards:
        running_total += reward
        cumulative.append(running_total)

    plot_directory.mkdir(parents=True, exist_ok=True)
    with (plot_directory / f"{label}_step_rewards.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", "reward", "cumulative_reward"))
        writer.writerows(zip(steps, rewards, cumulative, strict=True))

    write_line_svg(
        x_values=steps,
        series={"immediate reward": rewards},
        path=plot_directory / f"{label}_step_rewards.svg",
        title=f"Per-step rewards: {label}",
        x_label="Environment step",
        y_label="Immediate reward",
    )
    write_line_svg(
        x_values=steps,
        series={"cumulative reward": cumulative},
        path=plot_directory / f"{label}_cumulative_reward.svg",
        title=f"Cumulative reward: {label}",
        x_label="Environment step",
        y_label="Cumulative reward",
    )

