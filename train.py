"""Compatibility entry point; prefer python -m src.pipeline."""
from src.pipeline import main as train


if __name__ == "__main__":
    train()
