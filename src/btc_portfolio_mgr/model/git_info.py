"""Helpers to capture build provenance."""
from __future__ import annotations

import subprocess
from pathlib import Path


def current_git_sha(repo_root: Path) -> str:
    """Return the short HEAD SHA, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
