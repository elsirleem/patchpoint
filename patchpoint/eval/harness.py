"""Runs a retriever over a dataset split and writes results/<run_id>/."""

from __future__ import annotations

import json
from pathlib import Path

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
) -> Path:
    """Runs `retriever` (already indexed) over `split` of `dataset`.

    Writes config.json, predictions.jsonl, and metrics.json to results/<run_id>/.
    """
    examples = _split_examples(dataset, split)

    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        run_id=run_id,
        repo=dataset.repo,
        base_sha=examples[0].base_sha if examples else "",
        retriever=type(retriever).__name__,
        split=split,
        top_k=top_k,
    )
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

    per_example_recall: list[float] = []
    per_example_hit: list[float] = []

    with (run_dir / "predictions.jsonl").open("w") as f:
        for example in examples:
            ranked = retriever.query(
                f"{example.issue_title}\n\n{example.issue_body}", top_k=top_k
            )
            prediction = Prediction(issue_number=example.issue_number, ranked_files=ranked)
            f.write(prediction.model_dump_json() + "\n")

            per_example_recall.append(metrics.recall_at_k(ranked, example.gold_files, top_k))
            per_example_hit.append(metrics.hit_at_k(ranked, example.gold_files, top_k))

    summary = {
        "n": len(examples),
        f"recall@{top_k}": sum(per_example_recall) / len(per_example_recall) if examples else 0.0,
        f"hit@{top_k}": sum(per_example_hit) / len(per_example_hit) if examples else 0.0,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    return run_dir


def _split_examples(dataset: Dataset, split: str) -> list[IssueExample]:
    if split == "dev":
        return [e for e in dataset.examples if e.merged_at < dataset.dev_cutoff]
    if split == "test":
        return [e for e in dataset.examples if e.merged_at >= dataset.dev_cutoff]
    raise ValueError(f"unknown split {split!r}, expected 'dev' or 'test'")
