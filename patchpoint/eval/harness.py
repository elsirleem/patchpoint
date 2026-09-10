"""Runs a retriever over a dataset split and writes results/<run_id>/."""

from __future__ import annotations

import json
from pathlib import Path

from patchpoint.data.checkout import index_repo_at_sha
from patchpoint.eval import metrics
from patchpoint.retrievers.base import Retriever
from patchpoint.schemas import Dataset, IssueExample, Prediction, RunConfig

RESULTS_DIR = Path("results")


def run_eval(
    retriever: Retriever,
    dataset: Dataset,
    split: str,
    run_id: str,
    top_k: int = 10,
    limit: int | None = None,
) -> Path:
    """Runs `retriever` over `split` of `dataset`.

    Each example is indexed fresh at its own base_sha right before it's queried —
    a repo's file set differs across years of issues, so there's no single sha to
    index the whole split at once (see CLAUDE.md invariant 1). Writes config.json,
    predictions.jsonl, and metrics.json to results/<run_id>/.

    `limit`, when set, runs only the first `limit` examples (in the split's
    existing chronological order) — a cheap prototype for a retriever with real
    per-call cost, not a number to report. The caller is expected to have named
    `run_id` accordingly (see cli.py) so this can never be mistaken for the full,
    reportable result sitting next to it in results/.
    """
    examples = split_examples(dataset, split)
    if limit is not None:
        examples = examples[:limit]

    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        run_id=run_id,
        repo=dataset.repo,
        retriever=type(retriever).__name__,
        split=split,
        top_k=top_k,
        limit=limit,
    )
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

    per_example_recall: list[float] = []
    per_example_hit: list[float] = []

    with (run_dir / "predictions.jsonl").open("w") as f:
        for example in examples:
            files = index_repo_at_sha(dataset.repo, example.base_sha)
            retriever.index(files)
            ranked = retriever.query(
                f"{example.issue_title}\n\n{example.issue_body}", top_k=top_k
            )
            prediction = Prediction(issue_number=example.issue_number, ranked_files=ranked)
            f.write(prediction.model_dump_json() + "\n")

            per_example_recall.append(metrics.recall_at_k(ranked, example.gold_files, top_k))
            per_example_hit.append(metrics.hit_at_k(ranked, example.gold_files, top_k))

    summary: dict[str, float | int] = {
        "n": len(examples),
        f"recall@{top_k}": sum(per_example_recall) / len(per_example_recall) if examples else 0.0,
        f"hit@{top_k}": sum(per_example_hit) / len(per_example_hit) if examples else 0.0,
    }
    cost_usd = retriever.cost_usd()
    if cost_usd is not None:
        summary["cost_usd"] = cost_usd
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    return run_dir


def split_examples(dataset: Dataset, split: str) -> list[IssueExample]:
    if split == "dev":
        return [e for e in dataset.examples if e.merged_at < dataset.dev_cutoff]
    if split == "test":
        return [e for e in dataset.examples if e.merged_at >= dataset.dev_cutoff]
    raise ValueError(f"unknown split {split!r}, expected 'dev' or 'test'")
