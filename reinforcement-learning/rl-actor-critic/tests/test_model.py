import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_actor_critic.model import ActorCritic  # noqa: E402


class ActorCriticModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20)
        self.model = ActorCritic(observation_size=8, action_size=4, hidden_size=32)

    def test_batch_output_shapes(self) -> None:
        logits, values = self.model(torch.zeros(5, 8))
        self.assertEqual(logits.shape, (5, 4))
        self.assertEqual(values.shape, (5,))

    def test_single_deterministic_action(self) -> None:
        action, log_probability, entropy, value = self.model.choose_action(
            torch.zeros(8), deterministic=True
        )
        self.assertEqual(action.shape, ())
        self.assertEqual(log_probability.shape, ())
        self.assertEqual(entropy.shape, ())
        self.assertEqual(value.shape, ())
        self.assertIn(action.item(), range(4))


if __name__ == "__main__":
    unittest.main()

