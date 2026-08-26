"""Metric definitions.

Drafted by Claude at the user's request, overriding the default CLAUDE.md split
("I write these; you review only") for this instance — see the chat for the
pushback that preceded that decision. Review before trusting these numbers.
"""

from __future__ import annotations

from patchpoint.schemas import ScoredFile


def recall_at_k(ranked: list[ScoredFile], gold_files: list[str], k: int) -> float:
    """Fraction of the gold files that appear in the top k, for one issue.

    |top_k ∩ gold| / |gold|. Per-example — eval/harness.py macro-averages this
    across issues, so an issue with many gold files doesn't get more say in the
    final number than one with a single gold file.

    Truncates `ranked` to `k` itself rather than trusting the caller already did,
    so the function is correct in isolation, not just as harness.py currently
    calls it.
    """
    if not gold_files:
        raise ValueError("recall@k is undefined for an example with no gold files")

    top_k_paths = {sf.path for sf in ranked[:k]}
    return len(top_k_paths & set(gold_files)) / len(gold_files)


def hit_at_k(ranked: list[ScoredFile], gold_files: list[str], k: int) -> float:
    """1.0 if any gold file appears in the top k, else 0.0, for one issue.

    Unlike recall@k, this doesn't reward finding more of the gold set — it only
    asks whether a developer scanning the top k would land on at least one
    correct file. Standard in bug-localization work (often called "top-N
    accuracy" or "hit rate") because most issues in practice touch one file that
    matters and several incidental ones.
    """
    if not gold_files:
        raise ValueError("hit@k is undefined for an example with no gold files")

    top_k_paths = {sf.path for sf in ranked[:k]}
    return 1.0 if top_k_paths & set(gold_files) else 0.0
