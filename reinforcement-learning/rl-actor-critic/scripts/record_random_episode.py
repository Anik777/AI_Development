from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym

SEED=20

def main():
    project_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_directory = (
        project_root
        / "runs"
        / f"video_smoke_{timestamp}"
    )
    video_directory = run_directory / "videos"

    env = gym.make(
        "LunarLander-v3",
        render_mode="rgb_array",
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_directory),
        episode_trigger=lambda episode_number: True,
        name_prefix="random-policy",
        disable_logger=False,
    )

    observation, info = env.reset(seed=SEED)
    env.action_space.seed(SEED)

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    try:
        while not (terminated or truncated):
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            step_count += 1
    finally:
        # RecordVideo finishes encoding the MP4 during close().
        env.close()

    video_files = sorted(video_directory.glob("*.mp4"))

    print("Recording complete")
    print("------------------")
    print(f"Steps: {step_count}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Run directory: {run_directory}")

    for video_file in video_files:
        print(f"Video: {video_file}")


if __name__ == "__main__":
    main()