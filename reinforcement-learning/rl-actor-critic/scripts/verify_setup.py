#!/usr/bin/env python3
"""Verify the shared reinforcement-learning environment without training."""

from __future__ import annotations

import importlib.metadata
import sys

import gymnasium as gym
import numpy as np
import torch


def status(label: str, value: object) -> None:
    print(f"{label:<24} {value}")


def main() -> None:
    status("Python", sys.version.split()[0])
    status("NumPy", np.__version__)
    status("PyTorch", torch.__version__)
    status("Gymnasium", gym.__version__)
    status("MoviePy", importlib.metadata.version("moviepy"))
    status("CUDA compiled", torch.version.cuda)
    status("CUDA available", torch.cuda.is_available())
    if torch.cuda.is_available():
        status("CUDA device", torch.cuda.get_device_name(0))
        tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        status("CUDA calculation", (tensor.square()).tolist())

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    observation, _ = env.reset(seed=20)
    frame = env.render()
    _, reward, terminated, truncated, _ = env.step(0)
    env.close()
    status("Observation shape", observation.shape)
    status("Rendered frame", frame.shape)
    status("One-step reward", f"{reward:.3f}")
    status("Episode ended", terminated or truncated)
    print("\nSetup verification passed.")


if __name__ == "__main__":
    main()

