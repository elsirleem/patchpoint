"""Metric definitions — owned by the project author, not auto-generated.

Fill these in yourself; see CLAUDE.md: "I write these; you review only." Each
function should get its own test in tests/test_metrics.py (also yours to write).
"""

from __future__ import annotations

from patchpoint.schemas import ScoredFile


def recall_at_k(ranked: list[ScoredFile], gold_files: list[str], k: int) -> float:
    raise NotImplementedError("define recall@k yourself — see CLAUDE.md working agreement")


def hit_at_k(ranked: list[ScoredFile], gold_files: list[str], k: int) -> float:
    raise NotImplementedError("define hit@k yourself — see CLAUDE.md working agreement")
