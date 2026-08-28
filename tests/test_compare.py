from patchpoint.eval.compare import compare_runs
from patchpoint.schemas import Dataset, IssueExample, Prediction, RunConfig, ScoredFile


def _write_run(tmp_path, run_id: str, repo: str, split: str, predictions: list[Prediction]) -> None:
    run_dir = tmp_path / "results" / run_id
    run_dir.mkdir(parents=True)
    config = RunConfig(run_id=run_id, repo=repo, retriever="fake", split=split, top_k=10)
    (run_dir / "config.json").write_text(config.model_dump_json())
    with (run_dir / "predictions.jsonl").open("w") as f:
        for p in predictions:
            f.write(p.model_dump_json() + "\n")


def test_compare_runs_prefers_the_better_retriever(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    examples = [
        IssueExample(
            repo="a/b",
            issue_number=i,
            issue_title="t",
            issue_body="b",
            base_sha="abc",
            fix_sha="def",
            gold_files=["target.py"],
            merged_at="2024-01-01T00:00:00Z",
        )
        for i in range(1, 11)
    ]
    dataset = Dataset(repo="a/b", examples=examples, dev_cutoff="2025-01-01T00:00:00Z")
    (tmp_path / "data" / "datasets").mkdir(parents=True)
    (tmp_path / "data" / "datasets" / "a__b.json").write_text(dataset.model_dump_json())

    # "good" always finds the gold file; "bad" never does.
    good_predictions = [
        Prediction(issue_number=i, ranked_files=[ScoredFile(path="target.py", score=1.0)])
        for i in range(1, 11)
    ]
    bad_predictions = [
        Prediction(issue_number=i, ranked_files=[ScoredFile(path="other.py", score=1.0)])
        for i in range(1, 11)
    ]
    _write_run(tmp_path, "good", "a/b", "dev", good_predictions)
    _write_run(tmp_path, "bad", "a/b", "dev", bad_predictions)

    result = compare_runs("good", "bad", "recall@10")

    assert result["diff"] == 1.0
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.05
