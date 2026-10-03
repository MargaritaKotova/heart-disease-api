"""WSGI entry point: gunicorn src.app:app."""
from app.main import create_app

app = create_app()

__all__ = ["app"]
