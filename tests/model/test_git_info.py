from __future__ import annotations

from pathlib import Path

from btc_portfolio_mgr.model.git_info import current_git_sha


def test_current_git_sha_returns_string_in_repo() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    sha = current_git_sha(repo_root)
    # In a real repo, returns a short SHA (typically 7 chars)
    # If git fails for any reason, returns "unknown"
    assert isinstance(sha, str)
    assert len(sha) > 0


def test_current_git_sha_returns_unknown_outside_repo(tmp_path: Path) -> None:
    # tmp_path is not a git repo
    sha = current_git_sha(tmp_path)
    assert sha == "unknown"
