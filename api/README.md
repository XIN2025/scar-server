# First, make sure you have the development dependencies installed with UV:
uv pip install -e ".[dev]"

# Then you can run the tests using:
uv run pytest

# Running checks
uv run ruff check .