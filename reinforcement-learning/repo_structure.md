# Reinforcement-learning repository structure

```text
reinforcement-learning/
├── .venv/                         # shared generated Python environment
├── .gitignore
├── BOX2D_INSTALLATION.md          # Python 3.14/Box2D troubleshooting record
├── commands_run_guide.md          # personal command notes
├── repo_structure.md              # this repository map
├── requirements.txt               # shared runtime dependencies
├── requirements-notebook.txt      # optional Jupyter dependency
├── versions.txt                   # installed versions and verification journal
│
├── rl-actor-critic/
│   ├── configs/
│   │   └── lunar_lander.json      # normal hyperparameter defaults
│   ├── notebooks/
│   │   └── actor_critic_lunar_lander.ipynb
│   ├── scripts/
│   │   ├── evaluate.py            # replay a saved checkpoint
│   │   ├── inspect_environment.py # inspect Gymnasium spaces and one step
│   │   ├── random_agent.py        # reproducible random baseline
│   │   ├── record_random_episode.py
│   │   ├── run_training.sh        # Bash launcher and shell-level logs
│   │   ├── train.py               # configurable Python CLI
│   │   └── verify_setup.py        # dependency/GPU/environment check
│   ├── src/
│   │   └── rl_actor_critic/
│   │       ├── __init__.py
│   │       ├── config.py          # TrainingConfig and validation
│   │       ├── evaluation.py      # deterministic evaluation and MP4 recording
│   │       ├── model.py           # shared network, actor head, critic head
│   │       ├── run_io.py          # CSV, JSONL, .log, metadata, and SVG output
│   │       └── trainer.py         # rollout, GAE, losses, and optimization
│   ├── tests/
│   │   ├── test_config.py
│   │   ├── test_gae.py
│   │   └── test_model.py
│   ├── checkpoints/               # ignored legacy/generated output location
│   ├── logs/                      # ignored legacy/generated output location
│   ├── runs/                      # ignored complete per-run artifacts
│   ├── .gitignore
│   ├── README.md                  # complete run and algorithm guide
│   ├── installed_versions.txt     # pointer to ../versions.txt
│   ├── requirements-lock.txt      # lock-generation note
│   └── requirements.txt           # pointer to shared requirements
│
├── rl-continuous-control/         # future project
├── rl-fundamentals/               # future project
├── rl-policy-gradient/            # future project
└── rl-value-based/                # future project
```

Each actor-critic training execution creates a unique directory:

```text
rl-actor-critic/runs/<timestamp>_<name>/
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
│   ├── training_episode_rewards.svg     # raw and rolling reward per episode
│   ├── training_actor_loss.svg          # actor loss per episode
│   ├── training_critic_loss.svg         # critic loss per episode
│   ├── training_total_loss.svg          # combined objective per episode
│   └── evaluations/
│       ├── <label>_step_rewards.svg      # every immediate reward in the video
│       ├── <label>_cumulative_reward.svg
│       └── <label>_step_rewards.csv      # source data for both plots
├── command.txt                    # exact Python command
├── config.json                    # effective hyperparameters
├── events.jsonl                   # structured lifecycle/iteration events
├── iterations.log                 # one readable line per training iteration
├── metadata.json                  # system, package, CUDA, and device data
├── metrics.csv                    # one structured row per iteration
├── reward_curve.svg               # dependency-free reward visualization
├── summary.json                   # final result
└── training.log                   # progress, checkpoints, evaluations, errors
```

Runs started through `scripts/run_training.sh` additionally contain:

```text
shell_environment.log              # shell, command, Python/pip, and exit status
shell_console.log                  # complete stdout/stderr captured through tee
```

Hyperparameter precedence is:

```text
TrainingConfig fallback defaults
        ↓ overridden by
configs/lunar_lander.json
        ↓ overridden by
command-line options
        ↓ saved as
run/config.json
```

Video milestone selection is controlled by `record_episodes` in
`configs/lunar_lander.json` or `--record-episodes` on the command line. Episode
`0` means the untrained policy. The final policy is always recorded when video
recording is enabled.

