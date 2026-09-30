# Installation & Environment Setup

This repository can be run in a small `uv` environment. Training progress in `train_cdl_multi_dict.py` has tracked through `wandb`, which is therefore included as a regular runtime dependency.

## Prerequisites

- `uv` installed and available on `PATH`
- Python 3.12 available to `uv`
- Network access to PyPI and the PyTorch CUDA wheel index


## Standard Setup
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

Run commands *without* activating the environment (uv handles correct usage, this is best practice) by prefixing them with `uv run`, for example:

Alternatively, activate the environment directly:
```bash
source .venv/bin/activate
```
and then run python scripts regularly by prefixing with `python`.

If the `mrpro` package is intended to be used, e.g., to compute test metrics in an evalutation script of yours, clone the repository *next to* where this repository lives, then add the dependency here as per:
```bash
uv add --editable ../mrpro
```

## Cluster setup: pinned CUDA 12.6

- For the cluster setup: a sufficiently recent NVIDIA driver for CUDA 12.6

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
mrpro = { path = "../mrpro", editable = true }
```

Then resolve and install the environment:

```bash
uv lock
uv sync
```

`uv sync` creates `.venv` automatically. Use commands with `uv run` or activate the env explicitly as before.

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

The script uses relative paths such as `data/training/` and `data/validation/`, so it must be launched from the repo root unless those paths are changed.

