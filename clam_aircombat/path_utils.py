"""Path resolution utilities for external repositories."""

from __future__ import annotations

from pathlib import Path


class RepoNotFoundError(RuntimeError):
    """Raised when an expected repository path is missing."""


def resolve_repo_path(path: str | Path, repo_name: str) -> Path:
    repo_path = Path(path).expanduser().resolve()
    if not repo_path.exists():
        raise RepoNotFoundError(f"{repo_name} path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise RepoNotFoundError(f"{repo_name} path is not a directory: {repo_path}")
    return repo_path
