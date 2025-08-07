"""Expose FastAPI routers so main.py can include them via `app.routes.*`."""
from . import admin, agents, analytics, auth, chat, tasks, system, websocket_chat, generate, edge_agent  # noqa: F401

