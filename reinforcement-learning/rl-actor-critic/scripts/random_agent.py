#!/usr/bin/env python3
"""Run a reproducible random-policy baseline for one complete episode."""

import gymnasium as gym


SEED = 20


def main() -> None:
    env = gym.make("LunarLander-v3")
    observation, _ = env.reset(seed=SEED)
    env.action_space.seed(SEED)
    total_reward = 0.0
    action_counts = {action: 0 for action in range(env.action_space.n)}
    terminated = False
    truncated = False
    step_count = 0

    try:
        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            step_count += 1
            action_counts[action] += 1

            if step_count <= 5 or step_count % 25 == 0 or terminated or truncated:
                print(
                    f"step={step_count:4d} action={action} reward={reward:8.3f} "
                    f"total={total_reward:9.3f} y={observation[1]:7.3f}"
                )
    finally:
        env.close()

    ending = "terminated" if terminated else "truncated"
    print("\nEpisode summary")
    print("----------------")
    print(f"Seed: {SEED}")
    print(f"Steps: {step_count}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Ending: {ending}")
    print(f"Action counts: {action_counts}")


if __name__ == "__main__":
    main()

