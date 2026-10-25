"""Training configuration and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingConfig:
    """All tunable values used by training and evaluation."""

    env_id: str = "LunarLander-v3"
    episodes: int = 1_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    hidden_size: int = 128
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    max_steps_per_episode: int = 1_000
    seed: int = 20
    evaluation_seed: int = 2_026
    device: str = "auto"
    rolling_window: int = 100
    log_interval: int = 10
    checkpoint_interval: int = 100
    record_episodes: tuple[int, ...] = (0, 1, 100)
    record_video: bool = True
    deterministic_torch: bool = True
    solved_score: float = 200.0
    stop_on_solved: bool = False

    def validate(self) -> None:
        """Raise ValueError when a configuration cannot be trained safely."""

        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be between 0 and 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.value_loss_coef < 0.0:
            raise ValueError("value_loss_coef cannot be negative")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef cannot be negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.max_steps_per_episode < 1:
            raise ValueError("max_steps_per_episode must be positive")
        if self.rolling_window < 1:
            raise ValueError("rolling_window must be positive")
        if self.log_interval < 1:
            raise ValueError("log_interval must be positive")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if any(episode < 0 for episode in self.record_episodes):
            raise ValueError("record_episodes cannot contain negative values")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""

        data = asdict(self)
        data["record_episodes"] = list(self.record_episodes)
        return data

    def with_overrides(self, **overrides: Any) -> "TrainingConfig":
        """Return a validated copy with non-None override values."""

        clean = {key: value for key, value in overrides.items() if value is not None}
        if "record_episodes" in clean:
            clean["record_episodes"] = tuple(clean["record_episodes"])
        updated = replace(self, **clean)
        updated.validate()
        return updated

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        """Load configuration from a JSON file and reject unknown keys."""

        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        if "record_episodes" in data:
            data["record_episodes"] = tuple(data["record_episodes"])
        config = cls(**data)
        config.validate()
        return config

