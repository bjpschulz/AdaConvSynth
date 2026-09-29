# Multi-dictionary training environment

This repository can be run in a small `uv` environment. Two setup paths are documented below:

- a cluster setup pinned to the CUDA 12.6 PyTorch wheels;
- a standard setup where `uv` selects compatible PyTorch wheels for the platform.

Both setups install `wandb` as a regular runtime dependency because `train_cdl_multi_dict.py` imports and uses it.

## Prerequisites

- `uv` installed and available on `PATH`
- Python 3.12 available to `uv`
- For the cluster setup: a sufficiently recent NVIDIA driver for CUDA 12.6
- Network access to PyPI and the PyTorch CUDA wheel index

For the cluster setup, the PyTorch wheel includes the CUDA runtime libraries. A separate system CUDA toolkit is normally not required to run the training script, but the NVIDIA driver still needs to support CUDA 12.6.

## Cluster setup: pinned CUDA 12.6

From this directory:

```bash
cd /path/to/multi_dict_repo
uv init --bare
```

Add the following contents to the generated `pyproject.toml`:

```toml
[project]
name = "multi-dict-training"
version = "0.1.0"
description = "Multi-dictionary CDL-FISTA training"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch==2.14.0+cu126",
    "torchvision==0.29.0+cu126",
    "wandb>=0.30.0",
    "einops>=0.8",
    "matplotlib>=3.10",
    "numpy>=2.0",
]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }
```

Then resolve and install the environment:

```bash
uv lock
uv sync
```

`uv sync` creates `.venv` automatically. Activate it for commands that should use the environment directly:

```bash
source .venv/bin/activate
```

Alternatively, run commands without activating the environment by prefixing them with `uv run`.

## Standard setup: platform-selected PyTorch

If the machine does not require the cluster's exact CUDA 12.6 wheels, use the regular PyPI packages. In that case, `uv init` (without `--bare`) creates the normal project metadata, and `uv add` records each dependency in `pyproject.toml`:

```bash
cd /path/to/multi_dict_repo
uv init --python 3.12
uv add torch torchvision wandb einops matplotlib numpy
```

`uv add` resolves compatible versions for the current platform and creates or updates `uv.lock`. Install the resolved environment with:

```bash
uv sync
```

This setup does not guarantee the CUDA 12.6 Torch versions used by the cluster. If a particular Torch or CUDA build is required later, use the pinned cluster setup above instead.

## Verify PyTorch and CUDA

Run this before starting a long training job:

```bash
uv run python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

For the cluster setup, the expected Torch version is `2.14.0+cu126`, the expected CUDA runtime is `12.6`, and `CUDA available` should be `True` on a GPU allocation. For the standard setup, the Torch version and CUDA runtime depend on the platform and selected wheel. If CUDA is expected but `CUDA available` is `False`, check that the machine has a GPU, that `nvidia-smi` works, and that the NVIDIA driver is compatible with the installed wheel.

Check the remaining imports as well:

```bash
uv run python -c "import einops, matplotlib, numpy, torchvision, wandb; print('imports: ok')"
```

## Configure W&B

Log in once on the cluster, using the W&B instructions appropriate for your account:

```bash
uv run wandb login
```

For non-interactive cluster jobs, configure the API key through the cluster's secret mechanism or environment before starting the job. Do not commit the key to this repository.

## Run training

From the repository root, use:

```bash
uv run python train_cdl_multi_dict.py
```

The script uses relative paths such as `data/training/` and `data/validation/`, so it must be launched from `multi_dict_repo/` unless those paths are changed.

## Reproducing the environment

Commit `pyproject.toml` and `uv.lock` to version control after the initial setup. On another machine or cluster node, the environment can then be recreated with:

```bash
uv sync --locked
```

Do not commit `.venv/`, W&B credentials, or generated training outputs.
