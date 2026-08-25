"""Reciprocal Rank Fusion over any set of retrievers."""

from __future__ import annotations

from patchpoint.retrievers.base import Retriever
from patchpoint.schemas import ScoredFile


class RRFHybridRetriever(Retriever):
    def __init__(self, retrievers: list[Retriever], k: int = 60, pool_size: int = 100) -> None:
        if not retrievers:
            raise ValueError("RRFHybridRetriever needs at least one retriever")
        self.retrievers = retrievers
        self.k = k
        self.pool_size = pool_size

    def index(self, files: dict[str, str]) -> None:
        for retriever in self.retrievers:
            retriever.index(files)

    def query(self, issue_text: str, top_k: int = 10) -> list[ScoredFile]:
        fused: dict[str, float] = {}
        for retriever in self.retrievers:
            ranked = retriever.query(issue_text, top_k=self.pool_size)
            for rank, scored in enumerate(ranked, start=1):
                fused[scored.path] = fused.get(scored.path, 0.0) + 1.0 / (self.k + rank)

        ranked_fused = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [ScoredFile(path=p, score=s) for p, s in ranked_fused[:top_k]]
