"""Portable, non-identifying paths for human-facing reports."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> str:
    """Keep project-relative paths and Hub IDs; redact external local paths.

    This is for reports, not functional model/adapter references needed to reload.
    """
    path = Path(value)
    if not path.is_absolute() and ".." not in path.parts:
        return path.as_posix()
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return "<external path>"
