# Overview
This repository consists of two main scripts: `sft_pipeline.py` and `main.py`.
The former performs Supervised Fine Tuning on the `openai/gsm8k` dataset and
merges the result into a new model that is used as the starting point for GRPO
reinforcement learning. `main.py` trains the fine-tuned model via GRPO to
improve its reasoning abilities on `openai/gsm8k`.

# Dependencies
First install uv (https://docs.astral.sh/uv/getting-started/installation/)

Then, run `uv sync` to install dependencies. You may need to adjust PyTorch
versions based on your CUDA version. You can do this by editing
`pyproject.toml`, see (https://docs.astral.sh/uv/guides/integration/pytorch/) 
[https://docs.astral.sh/uv/guides/integration/pytorch/] for more information.

To run a script (for example, `main.py`), you can either do `uv run main.py` or 
`source .venv/bin/activate && python3 main.py`. The latter will activate the 
virtual environment in case you want to run other code samples.

# TensorBoard
To view metrics in realtime, you can run 
`tensorboard --logdir <output_path/logs> --port 6006` (with .venv acticvated)
from the root of this project to spin up a TensorBoard dashboard. If you're 
running the trainer on a remote server, you can port forward the dashboard by 
running `ssh -N -L 6006:localhost:6006 <remote-ip>` and opening a web browser 
to localhost:6006.

# Supervised Fine Tuning Pipeline
`sft_pipeline.py` is responsible for downloading a model from HuggingFace,
settting up LoRA, and performing SFT to teach the model the format of gsm8k
problems. This stage converts the answers from the `gsm8k` to the following
format:

- Chain-of-Thought reasoning is embedded in <think> tags
- Answers are wrapped in a \\box{} symbol.

`sft_pipeline.py` supports the following command-line arguments:

- `--output_dir`: The output directory to save the outputs (logs, configuration,
  model checkpoints, and final merged model)
- `--learning_rate`: The learning rate for SFT. Default is 2e-5. We found that
  an even smaller learning rate is necessary for fine tuning the
  `Qwen2.5-1.5B-Instruct`, in the range of 1e-6 to 5e-6.
- `--batch_size`: The maximum number of prompts to load on the GPU and train at
  a time. Defaultt is 32, though this requires a ton of memory. The actual batch
  size is calculated by `BATCH_SIZE * GRAD_ACCUM` to support memory-constrained
  training environments.
- `--grad_accum`: The number of batches to calculate gradients over before
  performing SGD on the loss. Default is 1. You can increase this parameter to compensate for
  lower batch sizes in memory-constrained environments.
- `--neftune_alpha`: Embedding noise coefficient. Default is 5.0. Larger values
  add more noise, 0 adds none. Adding embedding noise has been shown to improve
  fine-tuning performance (see [Jain et al.] (https://arxiv.org/pdf/2310.05914)).
- `--skip_baseline`: This will skip evaluating the baseline model on a test set of 
    `openai/gsm8k`. Default is False. Set to true if you want to try different
    hyperparameter configurations for SFT and don't want to re-evaluate the
    baseline performance.

Additionally, a few hyperparameters are statically set in the script itself, but
can be changed. These are as follows:
- `MODEL_ID`: The base model to download from HuggingFace. Default is 
    "Qwen/Qwen2.5-1.5B", but we also experimented with
    "Qwen/Qwen2.5-1.5B-Instruct".
- `MAX_SEQ_LEN`: The maximum output length before the generated response gets
  cut off. Default tis 768. Longer maximum outputs allow longer
  chains-of-though, but the longest answer in gsm8k is around 300-400 tokens.
- `SYSTEM_PROMPT`: The instruction prompt given to the base model.

# Reinforcement Learning Pipeline
## grpo.py
This file implements GRPO algorithm in PyTorch. It is configurable via
the GRPOConfig class, which provides parameters for the GRPO learning algorithm,
parameters passed to the LLM for controlling prompt outputs, and step values to
control when the algorithm logs metrics, saves checkpoint models, evaluates the
model on the training set, and the random seed.

By default, the GRPOTrainer receives a GRPOConfig that will train for a single
epoch (2000 time prompts) and evaluate + checkpoint every 50 timesteps. The
hyperparameters in this config correspond 1:1 with those passed to `main.py`,
although this class also provides additonal hyperparameters.

## main.py
`main.py` defines the main GRPO-based RL pipeline. It will load the merged SFT
model from the SFT pipeline and begin training it via GRPO. It supports the
following arguments:

- `--output_dir`: Path to save checpoinst, logs, and config. Defaul tis None,
  required.
- `--resume_from`: Path to a checkpoint model to continue training from. Default
  is None. Useful for continuing training from promising results.
- `--reward_think`: Reward value for correctly formatting the chain-of-thought
  response in a <think> tag. Default is 0.1.
- `--reward_box`: Reward value for correctly formatting the final answer in a
  \\box{}. Defaul tis 0.1.
- `--reward_correct`: Reward for answering the question correctly. Default is
  1.0.
- `--reward_incorrect`: Positive reward for answering the question incorrectly.
  Set this to a negative value to punish incorrect answers. Default is 0.0.

Additionally, a few hyperparameters are statically set in the script itself, but
can be changed. These are as follows:
- `SFT_MODEL_DIR`: Local path to the merged final output of the SFT pipeline.
  Default is `./qwen2.5-1.5b-sft_v4_base-merged`, but this is too large to 
  include in the repository. 
- `MAX_SEQ_LEN`: The maximum output length before the generated response gets
  cut off. Default tis 768. Longer maximum outputs allow longer
  chains-of-though, but the longest answer in gsm8k is around 300-400 tokens.
- `NUM_TRAIN`: The number of prompts per epoch. Default is 2000.
- `NUM_TEST`: The number of prompts to hold out for the evaluation set. Default
  is 500.
- `NUM_GENERATIONS`: The number of outputs to generate per prompt. Default is 6.
  Should be at least 4, though higher values require more compute & memory.
- `LEARNING_RATE`: The learning rate for GRPO. Default is 4e-6. 
- `BATCH_SIZE`: The maximum number of prompts to load on the GPU and train at
  a time. Default is 1. The actual batch size is calculated by 
  `BATCH_SIZE * GRAD_ACCUM` to support memory-constrained
  training environments.
- `GRAD_ACCUM`: The number of batches to calculate gradients over before
  performing SGD on the loss. Default is 1. You can increase this parameter to compensate for
  lower batch sizes in memory-constrained environments.
- `BETA`: Coefficient for the KL divergence penalty. Default is 0.04. High
  values penalize more, lower values allow the policy to drift more per
  timestep.
- `MAX_GRAD_NORM`: Clipping threshold for the gradient. Default is 1.
- `SYSTEM_PROMPT`: The instruction prompt given to the base model.

# IMPORTANT
This GRPO pipeline depends on the SFT trained version of Qwen2.5-1.5b. Make sure
you train a fine-tuned model using the `sft_pipeline.py` script, and point the
`SFT_MODEL_DIR` param in `main.py` to the relative path of this model dir.

