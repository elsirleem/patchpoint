"""Paired bootstrap significance test — owned by the project author.

Every "X beats Y" claim in this project must go through this function, not two
overlapping confidence intervals (see CLAUDE.md invariant 5). Fill it in yourself.
"""

from __future__ import annotations


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Return at least {"diff": ..., "ci_low": ..., "ci_high": ..., "p_value": ...}."""
    raise NotImplementedError("define the paired bootstrap test yourself — see CLAUDE.md")
