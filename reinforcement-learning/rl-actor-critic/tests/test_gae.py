import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_actor_critic.trainer import compute_gae  # noqa: E402


class GeneralizedAdvantageTests(unittest.TestCase):
    def test_two_step_undiscounted_returns(self) -> None:
        advantages, returns = compute_gae(
            rewards=[1.0, 1.0],
            values=[torch.tensor(0.5), torch.tensor(0.25)],
            bootstrap_masks=[1.0, 0.0],
            last_value=torch.tensor(0.0),
            gamma=1.0,
            gae_lambda=1.0,
        )
        torch.testing.assert_close(advantages, torch.tensor([1.5, 0.75]))
        torch.testing.assert_close(returns, torch.tensor([2.0, 1.0]))


if __name__ == "__main__":
    unittest.main()

