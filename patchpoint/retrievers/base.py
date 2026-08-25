"""The interface every retriever variant implements.

New retrievers implement this and nothing else changes — see CLAUDE.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from patchpoint.schemas import ScoredFile


class Retriever(ABC):
    @abstractmethod
    def index(self, files: dict[str, str]) -> None:
        """Build an index over `files` (path -> file content) at a fixed base_sha."""

    @abstractmethod
    def query(self, issue_text: str, top_k: int = 10) -> list[ScoredFile]:
        """Return up to `top_k` files ranked by likelihood of needing a change."""
