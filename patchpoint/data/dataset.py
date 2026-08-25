"""Builds a Dataset (examples + chronological split) from mined GitHub examples."""

from __future__ import annotations

from patchpoint.data.github import mine_examples
from patchpoint.schemas import Dataset


def build_dataset(repo: str, max_examples: int = 200, dev_fraction: float = 0.8) -> Dataset:
    """Mines examples and marks the chronological dev/test cutoff.

    The oldest `dev_fraction` of examples (by merged_at) go to dev; the rest go to
    test. The test split must only be read once, at the end — see CLAUDE.md
    invariant 3.
    """
    examples = mine_examples(repo, max_examples=max_examples)
    examples.sort(key=lambda e: e.merged_at)

    if not examples:
        return Dataset(repo=repo, examples=[], dev_cutoff="")

    cutoff_idx = max(1, int(len(examples) * dev_fraction)) - 1
    dev_cutoff = examples[cutoff_idx].merged_at

    return Dataset(repo=repo, examples=examples, dev_cutoff=dev_cutoff)
