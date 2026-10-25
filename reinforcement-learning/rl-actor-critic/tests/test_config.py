import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_actor_critic.config import TrainingConfig  # noqa: E402


class TrainingConfigTests(unittest.TestCase):
    def test_command_line_style_overrides(self) -> None:
        config = TrainingConfig().with_overrides(
            episodes=12, gae_lambda=0.8, record_episodes=[0, 1, 12]
        )
        self.assertEqual(config.episodes, 12)
        self.assertEqual(config.gae_lambda, 0.8)
        self.assertEqual(config.record_episodes, (0, 1, 12))

    def test_invalid_lambda_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "gae_lambda"):
            TrainingConfig(gae_lambda=1.1).validate()

    def test_json_loading_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"episodes": 3, "mystery": 1}))
            with self.assertRaisesRegex(ValueError, "mystery"):
                TrainingConfig.from_json(path)


if __name__ == "__main__":
    unittest.main()

