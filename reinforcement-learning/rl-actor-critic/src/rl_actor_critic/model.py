"""Neural-network actor and critic."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """A shared feature network with separate policy and value heads."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.features = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_size, action_size)
        self.critic_head = nn.Linear(hidden_size, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.features:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and scalar state values."""

        features = self.features(observations)
        logits = self.actor_head(features)
        values = self.critic_head(features).squeeze(-1)
        return logits, values

    def distribution_and_value(
        self, observations: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        logits, values = self(observations)
        return Categorical(logits=logits), values

    def choose_action(
        self, observations: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose actions and return action, log-probability, entropy, and value."""

        distribution, values = self.distribution_and_value(observations)
        actions = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        return (
            actions,
            distribution.log_prob(actions),
            distribution.entropy(),
            values,
        )

