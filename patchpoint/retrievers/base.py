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

    def cost_usd(self) -> float | None:
        """Real (not estimated) USD cost of API calls made by this instance so far.

        None for retrievers with no external API cost (the default). A retriever
        that does incur cost overrides this — see retrievers/agentic.py.
        """
        return None
