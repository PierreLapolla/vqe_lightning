![Ruff](https://img.shields.io/badge/ruff-enabled-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

# Python App Template

This project provides a starter structure and tooling for Python apps, aiming for a consistent and modern dev
experience.

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
