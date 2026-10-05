"""FastAPI backend: REST commands, WebSocket state stream, static teach-pendant UI."""

from .app import create_app

__all__ = ["create_app"]
