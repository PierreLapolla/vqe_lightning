![Ruff](https://img.shields.io/badge/ruff-enabled-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

# ADAPT-VQE Lightning

A PyTorch Lightning + PennyLane project for training a variational quantum eigensolver (VQE) with Lightning backends, including configurable toy Hamiltonians, checkpointing, and optional Weights & Biases logging.

## Installation

### Requirements

- [UV](https://docs.astral.sh/uv/) package manager

### Clone the repository

```bash
  git clone https://github.com/PierreLapolla/lightning_template.git
  cd python_template
```

### Initialize your environment

```bash
  uv sync
  uv run pre-commit install
```

## Running the project

To run the project locally, run the following command:

```bash
  uv run -m src.app
```

## Tests, linting and formatting

```bash
  uv run pytest
```

```bash
  uvx ruff check . --fix
```

```bash
  uvx ruff format .
```

Run all hooks manually:

```bash
  uv run pre-commit run --all-files
```

## License

This project is licensed under the MIT [LICENSE](LICENSE)
