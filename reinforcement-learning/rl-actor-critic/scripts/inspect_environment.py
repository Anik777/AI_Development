import gymnasium as gym
print("Gymnasium version:", gym.__version__)
env = gym.make("LunarLander-v3")
print(env)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
observation, info = env.reset(seed=20)
print("Initial Observation :", observation)
print("Observation shape :", observation.shape)
print("Reset Information :", info)

env.action_space.seed(20)
action = env.action_space.sample()
print("Selected Action :", action)

next_observation, reward, terminated, truncated, info = env.step(action)

print("Next observation:", next_observation)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

env.close()