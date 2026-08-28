"""Compares two results/ runs with the paired bootstrap test.

Reads saved predictions rather than re-running retrieval — see CLAUDE.md:
"Analysis reads saved predictions. Never re-run retrieval to answer an analysis
question."
"""

from __future__ import annotations

from pathlib import Path

from patchpoint.eval import metrics
from patchpoint.eval.bootstrap import paired_bootstrap
from patchpoint.eval.harness import RESULTS_DIR, split_examples
from patchpoint.schemas import Dataset, Prediction, RunConfig

DATASETS_DIR = Path("data") / "datasets"

_METRIC_FNS = {
    "recall": metrics.recall_at_k,
    "hit": metrics.hit_at_k,
}


def _parse_metric(metric: str) -> tuple[str, int]:
    name, _, k_str = metric.partition("@")
    if name not in _METRIC_FNS or not k_str.isdigit():
        raise ValueError(f"unknown metric {metric!r}, expected e.g. 'recall@10' or 'hit@10'")
    return name, int(k_str)


def _per_example_scores(run_id: str, metric: str) -> dict[int, float]:
    run_dir = RESULTS_DIR / run_id
    config = RunConfig.model_validate_json((run_dir / "config.json").read_text())
    dataset = Dataset.model_validate_json(
        (DATASETS_DIR / f"{config.repo.replace('/', '__')}.json").read_text()
    )
    gold_by_issue = {e.issue_number: e.gold_files for e in split_examples(dataset, config.split)}

    metric_name, k = _parse_metric(metric)
    metric_fn = _METRIC_FNS[metric_name]

    scores: dict[int, float] = {}
    with (run_dir / "predictions.jsonl").open() as f:
        for line in f:
            prediction = Prediction.model_validate_json(line)
            gold = gold_by_issue.get(prediction.issue_number)
            if gold is None:
                continue
            scores[prediction.issue_number] = metric_fn(prediction.ranked_files, gold, k)

    return scores


def compare_runs(run_id_a: str, run_id_b: str, metric: str) -> dict[str, float]:
    """Paired bootstrap comparison of two results/ runs on the same metric.

    Both runs must have evaluated the same split of the same dataset — scores are
    aligned by issue_number, and any issue missing from either run is dropped.
    """
    scores_a = _per_example_scores(run_id_a, metric)
    scores_b = _per_example_scores(run_id_b, metric)

    shared_issues = sorted(scores_a.keys() & scores_b.keys())
    if not shared_issues:
        raise ValueError(f"no examples in common between {run_id_a!r} and {run_id_b!r}")

    return paired_bootstrap(
        [scores_a[i] for i in shared_issues],
        [scores_b[i] for i in shared_issues],
    )
