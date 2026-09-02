"""Checks out a repo's file contents at a specific commit, for indexing.

Uses a local bare clone (cached under .cache/repos/) rather than the GitHub API:
an eval run indexes at a different base_sha for nearly every example, and doing
that through the API would mean hundreds of requests per example. `git archive`
gets a full file tree in one shot; results are cached per (repo, sha) so re-runs
across retrievers don't repeat the work.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from patchpoint.data.cache import cached_json

REPOS_DIR = Path(".cache") / "repos"
_MAX_FILE_BYTES = 1_000_000  # skip pathologically large files (generated assets, etc.)


def _bare_clone_path(repo: str) -> Path:
    return REPOS_DIR / f"{repo.replace('/', '__')}.git"


def _ensure_cloned(repo: str) -> Path:
    path = _bare_clone_path(repo)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(path)],
            check=True,
            capture_output=True,
        )
    return path


def _archive_at(repo_path: Path, sha: str) -> bytes:
    """git archive, fetching first if `sha` isn't in the local clone yet.

    Handles two cases: a sha newer than the last clone/fetch (e.g. the live
    HEAD, via resolve_head_sha), and — defensively — any other sha the initial
    clone didn't happen to include.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_path), "archive", sha], check=False, capture_output=True
    )
    if result.returncode != 0:
        subprocess.run(
            ["git", "-C", str(repo_path), "fetch", "origin", sha],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "-C", str(repo_path), "archive", sha], check=True, capture_output=True
        )
    return result.stdout


def resolve_head_sha(repo: str) -> str:
    """Returns `repo`'s current default-branch HEAD commit sha on GitHub.

    Not cached — unlike a fixed base_sha, HEAD is time-varying by definition, so
    caching it would silently serve a stale commit.
    """
    repo_path = _ensure_cloned(repo)
    subprocess.run(
        ["git", "-C", str(repo_path), "fetch", "origin", "HEAD"], check=True, capture_output=True
    )
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "FETCH_HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@cached_json("index_files")
def index_repo_at_sha(repo: str, sha: str) -> dict[str, str]:
    """Returns {path: content} for every text file in `repo` at `sha`."""
    repo_path = _ensure_cloned(repo)

    with tempfile.TemporaryDirectory() as tmp:
        archive = _archive_at(repo_path, sha)
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)

        files: dict[str, str] = {}
        for path in Path(tmp).rglob("*"):
            if not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
                continue
            try:
                files[str(path.relative_to(tmp))] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue  # binary file

    return files
