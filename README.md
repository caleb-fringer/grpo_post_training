# IMPORTANT
This GRPO pipeline depends on the SFT trained version of Qwen2.5-1.5b-sft-math.
Please download it in the root directory of this codebase, and make sure you set
the `SFT_MODEL_DIR` param in `main.py` to the relative path of this model dir.

# Instructions
To run, first install uv (https://docs.astral.sh/uv/getting-started/installation/)

Then, run `uv sync` to install dependencies. You may need to adjust PyTorch
versions based on your CUDA version. You can do this by editing
`pyproject.toml`, see https://docs.astral.sh/uv/guides/integration/pytorch/ for
more information.

To run, you can either do `uv run main` or `source .venv/bin/activate && python3
main.py`. The latter will activate the virtual environment in case you want to
run other code samples.

# TensorBoard
To view metrics in realtime, you can run 
`tensorboard --logdir grpo_output_local --port 6006` from the root of this 
project to spin up a TensorBoard dashboard. If you're running the trainer
on a remote server, you can port forward the dashboard by running 
`ssh -N -L 6006:localhost:6006 <remote-ip>` and opening a web browser to
localhost:6006.
