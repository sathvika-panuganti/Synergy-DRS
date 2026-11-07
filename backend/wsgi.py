"""WSGI entry point for production hosts such as Render or Gunicorn."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import app  # noqa: E402
