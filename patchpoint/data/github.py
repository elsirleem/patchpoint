"""Mines (issue, fix) example pairs from a GitHub repo's history.

A PR counts as a fix when its body references an issue with a closing keyword
(Fixes/Closes/Resolves #N) and it was merged. The fix's parent commit is base_sha;
the files that PR touched are gold_files. All API calls are cached on disk (see
patchpoint.data.cache) so a re-run costs nothing.

Written but unrun as of Day 0 — see CLAUDE.md status.
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

from patchpoint.data.cache import cached_json
from patchpoint.schemas import IssueExample

API_ROOT = "https://api.github.com"
_CLOSES_RE = re.compile(
    r"\b(?:close[sd]?|fixe?[sd]?|resolve[sd]?)\s*:?\s*#(\d+)", re.IGNORECASE
)


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@cached_json("github_prs")
def _list_merged_prs(repo: str, page: int) -> list[dict[str, Any]]:
    resp = httpx.get(
        f"{API_ROOT}/repos/{repo}/pulls",
        params={
            "state": "closed",
            "per_page": 100,
            "page": page,
            "sort": "updated",
            "direction": "asc",
        },
        headers=_headers(),
        timeout=30.0,
    )
    resp.raise_for_status()
    return [pr for pr in resp.json() if pr.get("merged_at")]


@cached_json("github_pr_files")
def _pr_files(repo: str, pr_number: int) -> list[str]:
    resp = httpx.get(
        f"{API_ROOT}/repos/{repo}/pulls/{pr_number}/files",
        params={"per_page": 100},
        headers=_headers(),
        timeout=30.0,
    )
    resp.raise_for_status()
    return [f["filename"] for f in resp.json()]


@cached_json("github_issue")
def _issue(repo: str, issue_number: int) -> dict[str, Any]:
    resp = httpx.get(
        f"{API_ROOT}/repos/{repo}/issues/{issue_number}", headers=_headers(), timeout=30.0
    )
    resp.raise_for_status()
    return resp.json()


@cached_json("github_commit_parent")
def _commit_parent(repo: str, sha: str) -> str:
    resp = httpx.get(f"{API_ROOT}/repos/{repo}/commits/{sha}", headers=_headers(), timeout=30.0)
    resp.raise_for_status()
    parents = resp.json()["parents"]
    if not parents:
        raise ValueError(f"{sha} in {repo} has no parent commit")
    return str(parents[0]["sha"])


def mine_examples(repo: str, max_examples: int = 200) -> list[IssueExample]:
    """Walks merged PRs oldest-first, keeping the ones that close an issue."""
    examples: list[IssueExample] = []
    page = 1

    while len(examples) < max_examples:
        prs = _list_merged_prs(repo, page)
        if not prs:
            break
        page += 1

        for pr in prs:
            issue_numbers = _CLOSES_RE.findall(pr.get("body") or "")
            if not issue_numbers:
                continue

            fix_sha = pr.get("merge_commit_sha")
            if not fix_sha:
                continue

            try:
                gold_files = _pr_files(repo, pr["number"])
                if not gold_files:
                    continue

                base_sha = _commit_parent(repo, fix_sha)
                issue = _issue(repo, int(issue_numbers[0]))
            except httpx.HTTPStatusError:
                # E.g. a merge commit that's no longer reachable after history was
                # rewritten (squash/rebase), or an issue that was since deleted.
                # Real git history has this kind of noise; skip the PR, don't abort.
                continue

            examples.append(
                IssueExample(
                    repo=repo,
                    issue_number=issue["number"],
                    issue_title=issue["title"],
                    issue_body=issue.get("body") or "",
                    base_sha=base_sha,
                    fix_sha=fix_sha,
                    gold_files=gold_files,
                    merged_at=pr["merged_at"],
                )
            )
            if len(examples) >= max_examples:
                break

    return examples
