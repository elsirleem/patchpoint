"""Paired bootstrap significance test.

Drafted by Claude at the user's request, overriding the default CLAUDE.md split
("I write these; you review only") for this instance — see the chat for the
pushback that preceded that decision. Review before trusting these numbers.

Every "X beats Y" claim in this project must go through this function, not two
overlapping confidence intervals (see CLAUDE.md invariant 5) — overlapping CIs
can hide a real, consistent per-example difference, because the width of each
CI is driven by variance *across issues*, not by how consistently A beats B on
the same issues.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap test for whether A's mean score differs from B's.

    scores_a and scores_b must be the same per-example metric (e.g. recall@10),
    aligned index-for-index over the same examples in the same order. "Paired"
    means each bootstrap resample redraws *examples* (with replacement) and
    recomputes both means on that same resample — so within-example correlation
    between A and B (an easy issue is easy for both) is preserved, rather than
    resampling A's and B's scores independently, which would inflate the
    apparent uncertainty in the diff.

    Returns:
        diff: mean(scores_a) - mean(scores_b) on the real (non-resampled) data.
        ci_low, ci_high: 95% percentile bootstrap CI on that diff.
        p_value: two-sided. Twice the smaller tail of resampled diffs that
            crossed zero, capped at 1.0 — i.e. how often resampling flips which
            system looks better. A small p means the *sign* of diff is stable
            under resampling, which is a claim about consistency, not effect size.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must be paired — same length, same examples")
    if not scores_a:
        raise ValueError("need at least one paired example")

    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    n = len(a)

    observed_diff = float(a.mean() - b.mean())

    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_diffs = a[resample_idx].mean(axis=1) - b[resample_idx].mean(axis=1)

    ci_low, ci_high = np.percentile(resampled_diffs, [2.5, 97.5])
    tail = min(float(np.mean(resampled_diffs <= 0)), float(np.mean(resampled_diffs >= 0)))
    p_value = min(1.0, 2 * tail)

    return {
        "diff": observed_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
    }
