"""Lazy / windowed I/O helpers for large drone rasters and point clouds.

Anything that needs to *open* a multi-GB raster or LAS file should funnel
through here so we never accidentally read a whole product into memory.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_path"]


def resolve_path(p: str | Path) -> Path:
    """Resolve a path and fail loudly if it doesn't exist.

    Per project rule: no silent fallbacks. If the file is missing, raise.
    """
    path = Path(p).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Expected input not found: {path}")
    return path
