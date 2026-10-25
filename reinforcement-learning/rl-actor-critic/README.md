# LunarLander actor–critic

This project trains a discrete actor–critic agent on Gymnasium's
`LunarLander-v3`. It uses one complete episode per optimizer iteration and
Generalized Advantage Estimation (GAE) to control the bias–variance tradeoff.

The command-line program and Jupyter notebook call the same implementation in
`src/rl_actor_critic`; the notebook is not a separate copy of the algorithm.

## Activate the shared environment

From the reinforcement-learning directory:

```bash
cd /data/AI_Development/reinforcement-learning
source .venv/bin/activate
```

Verify the installation:

```bash
python rl-actor-critic/scripts/verify_setup.py
```

The Python 3.14 Box2D installation workaround is documented in
`../BOX2D_INSTALLATION.md`.

## Run a short training experiment

```bash
python rl-actor-critic/scripts/train.py \
  --episodes 10 \
  --device cpu \
  --record-episodes 0 1 10 \
  --run-name first-test
```

This is a pipeline check, not enough training to solve LunarLander.

## Run the default experiment

```bash
python rl-actor-critic/scripts/train.py
```

Defaults are read from `configs/lunar_lander.json`. The default device is
`auto`, which selects the GPU when CUDA is available. For this small network,
CPU can sometimes be faster because the simulator itself runs on the CPU.

## Override hyperparameters

Command-line values override the JSON configuration without modifying it:

```bash
python rl-actor-critic/scripts/train.py \
  --episodes 1500 \
  --gae-lambda 0.90 \
  --gamma 0.99 \
  --learning-rate 0.0002 \
  --hidden-size 256 \
  --entropy-coef 0.005 \
  --value-loss-coef 0.5 \
  --max-grad-norm 0.5 \
  --rolling-window 100 \
  --log-interval 5 \
  --checkpoint-interval 50 \
  --record-episodes 0 1 100 500 1000 \
  --device cuda \
  --run-name lambda-090
```

List every option:

```bash
python rl-actor-critic/scripts/train.py --help
```

There are three places to control hyperparameters, in precedence order:

1. **CLI overrides** in the command you run, such as `--episodes 500` or
   `--gae-lambda 0.90`. These have the highest priority and affect one run.
2. **Normal experiment defaults** in `configs/lunar_lander.json`. Edit this
   file when you want future runs to share the same settings.
3. **Code-level fallback defaults** in the `TrainingConfig` class in
   `src/rl_actor_critic/config.py`. Usually you should leave these unchanged.

Every run writes its final effective values to `config.json`, so you can
always determine exactly which hyperparameters produced a checkpoint.

Important controls:

- `--episodes`: number of training episodes and optimizer iterations.
- `--gae-lambda`: GAE parameter in `[0, 1]`; lower values reduce variance but
  add bootstrap bias.
- `--gamma`: future-reward discount factor.
- `--learning-rate`: Adam optimizer step size.
- `--hidden-size`: width of each shared hidden layer.
- `--entropy-coef`: exploration bonus strength.
- `--value-loss-coef`: critic loss weight.
- `--record-episodes`: evaluation/video milestones. Episode `0` is the
  completely untrained actor.
- `--no-video`: keep evaluations but skip MP4 encoding.
- `--stop-on-solved`: stop once the rolling reward reaches `--solved-score`.

The final policy is always evaluated, checkpointed, and—unless `--no-video`
is used—recorded.

### Select exactly which episodes to record

In `configs/lunar_lander.json`, edit:

```json
"record_episodes": [0, 1, 100, 500]
```

Or override it for one run:

```bash
python rl-actor-critic/scripts/train.py \
  --episodes 1000 \
  --record-episodes 0 1 25 100 500
```

`0` records the untrained model. Each positive number records the model after
that training episode. The final model is recorded automatically, so episode
1000 does not need to appear in the list when training for 1000 episodes.

## Run through Bash

The Bash launcher records shell-level provenance in addition to Python logs:

```bash
bash rl-actor-critic/scripts/run_training.sh \
  --episodes 300 \
  --gae-lambda 0.95 \
  --record-episodes 0 1 100 300 \
  --run-name bash-example
```

The launcher demonstrates:

- strict mode with `set -Eeuo pipefail`;
- Bash arrays for safely preserving arguments;
- `$@` for forwarding command-line arguments;
- `tee` for simultaneous terminal and file output;
- `${PIPESTATUS[0]}` for preserving the Python program's exit status.

## Per-run artifacts

Every execution creates a unique directory under `runs/`:

```text
runs/<timestamp>_<name>/
├── checkpoints/
│   ├── episode_0001.pt
│   ├── episode_0100.pt
│   └── final.pt
├── videos/
│   ├── evaluation_episode_0000-episode-0.mp4
│   ├── evaluation_episode_0001-episode-0.mp4
│   ├── evaluation_episode_0100-episode-0.mp4
│   └── evaluation_final-episode-0.mp4
├── plots/
│   ├── training_episode_rewards.svg
│   ├── training_actor_loss.svg
│   ├── training_critic_loss.svg
│   ├── training_total_loss.svg
│   └── evaluations/
│       ├── episode_0000_step_rewards.svg
│       ├── episode_0000_cumulative_reward.svg
│       ├── episode_0000_step_rewards.csv
│       └── ...                    # each recorded milestone and final
├── command.txt
├── config.json
├── events.jsonl
├── iterations.log
├── metadata.json
├── metrics.csv
├── reward_curve.svg
├── summary.json
└── training.log
```

When the Bash launcher is used, it also creates:

```text
shell_environment.log
shell_console.log
```

Logging behavior:

- `metrics.csv` contains one immediately flushed row for every training
  iteration/episode.
- `events.jsonl` records run lifecycle, every iteration, and every evaluation.
- `iterations.log` contains one immediately flushed, human-readable line for
  every training iteration, regardless of `--log-interval`.
- `training.log` is created for every run and records startup, periodic
  progress, checkpoints, evaluations, completion, and exception tracebacks.
  Its routine progress frequency is controlled by `--log-interval`.
- `config.json` stores the effective configuration after CLI overrides.
- `metadata.json` stores Python, package, platform, CUDA, and device details.
- `summary.json` stores final performance and elapsed time.
- `plots/training_episode_rewards.svg` shows episode and rolling rewards.
- The three training loss plots keep actor, critic, and total loss separate so
  the critic's larger numeric scale does not hide the actor curve.
- `plots/evaluations/<label>_step_rewards.svg` plots every immediate reward
  from the exact fixed-seed evaluation used for that checkpoint's video.
- Each evaluation also gets a cumulative-reward SVG and a step-level CSV.

`runs/latest_run.txt` points to the most recently started run.

## Compare videos fairly

Training uses stochastic policy samples. Checkpoint videos instead use:

1. a separate evaluation environment;
2. the same `evaluation_seed` at every milestone;
3. greedy actions (`argmax` of actor probabilities);
4. no gradient or optimizer updates.

Consequently, visual changes between episode 1, episode 100, and final come
from learned policy changes rather than a different initial lander state.
The matching per-step reward plots use those same evaluation trajectories, so
the video and plot always describe the same policy, seed, and episode.

## Algorithm

The model has a shared two-layer feature network and two output heads:

- actor logits define `Categorical(logits=...)` over four discrete actions;
- the critic estimates the scalar state value `V(s)`.

For a transition, the temporal-difference residual is

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

GAE computes advantages backward through an episode:

```text
A_t = delta_t + gamma * lambda * A_{t+1}
```

The optimizer minimizes

```text
actor_loss  = -mean(log pi(a_t | s_t) * normalized_advantage_t)
critic_loss = 0.5 * mean((return_t - V(s_t))^2)
total_loss  = actor_loss
            + value_loss_coef * critic_loss
            - entropy_coef * entropy
```

Terminated episodes do not bootstrap beyond their final state. Truncated
episodes bootstrap from the critic because truncation is an external limit,
not necessarily a terminal state in the underlying task.

## Evaluate an existing checkpoint

Without recording:

```bash
python rl-actor-critic/scripts/evaluate.py \
  rl-actor-critic/runs/<run>/checkpoints/final.pt \
  --device auto
```

With a new video:

```bash
python rl-actor-critic/scripts/evaluate.py \
  rl-actor-critic/runs/<run>/checkpoints/final.pt \
  --video-dir rl-actor-critic/runs/<run>/videos/manual \
  --label replay
```

## Tests

```bash
python -m unittest discover -s rl-actor-critic/tests -v
```

The tests cover configuration validation, actor/critic tensor shapes, action
selection, and a known two-step GAE calculation.

## Jupyter notebook

Install JupyterLab into the already configured shared environment if it is not
present:

```bash
python -m pip install jupyterlab
```

Start it from the reinforcement-learning directory:

```bash
python -m jupyter lab
```

Open `rl-actor-critic/notebooks/actor_critic_lunar_lander.ipynb`. Change the
`TrainingConfig(...)` values in its configuration cell, then run the training
cell. It invokes the same `train()` function as the CLI.

