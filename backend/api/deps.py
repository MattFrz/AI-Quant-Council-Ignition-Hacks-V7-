"""Shared FastAPI dependencies. Step 3.5.

Every route function that needs config or other request-scoped context
should import from here rather than reaching into backend.config directly -
keeps the dependency graph in one place so it's obvious what each route
actually touches. Add more shared dependencies here as later routes
(job lookups, auth, etc.) need them - don't scatter fresh Depends() targets
across individual route files.
"""
from __future__ import annotations

from backend.config import Settings, settings


def get_settings() -> Settings:
    return settings
