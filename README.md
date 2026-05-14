# atic-cbfs

Differential-drive planning with CasADi.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# create the venv and install dependencies from pyproject.toml
uv sync
```

## Running

```bash
uv run python expert_data_generation/planner.py
uv run python expert_data_generation/model.py
uv run python learning-cbf/train.py
```

Or activate the venv directly:

```bash
source .venv/bin/activate
python expert_data_generation/planner.py
```

## Dependencies
Add new packages to `pyproject.toml`with `uv add <pkg>`.
