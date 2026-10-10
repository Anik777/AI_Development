#!/bin/bash
# Reinforcement Learning
export AI_HOME="/data/AI_Development"
mkdir -p "$AI_HOME"/reinforcement-learning/{rl-fundamentals,rl-value-based,rl-policy-gradient,rl-actor-critic,rl-continuous-control}

# Computer Vision
mkdir -p "$AI_HOME"/computer-vision/{image-processing,image-classification,object-detection,image-segmentation,vision-transformers}

# Language Models
mkdir -p "$AI_HOME"/language-models/{nlp-fundamentals,transformer-from-scratch,llm-pretraining,llm-finetuning,llm-inference}

# Shared Resources
mkdir -p "$AI_HOME"/shared/{datasets,pretrained-models,documentation}

# Experiments
mkdir -p "$AI_HOME"/experiments/{tensorboard,benchmarks}
