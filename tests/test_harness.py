import json

from patchpoint.eval import harness, metrics
from patchpoint.retrievers.bm25 import BM25Retriever
from patchpoint.schemas import Dataset, IssueExample


def test_run_eval_writes_config_predictions_and_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(metrics, "recall_at_k", lambda ranked, gold, k: 1.0)
    monkeypatch.setattr(metrics, "hit_at_k", lambda ranked, gold, k: 1.0)
    monkeypatch.setattr(
        harness, "index_repo_at_sha", lambda repo, sha: {"auth.py": "def login(): pass"}
    )

    example = IssueExample(
        repo="a/b",
        issue_number=1,
        issue_title="login broken",
        issue_body="auth session fails",
        base_sha="abc",
        fix_sha="def",
        gold_files=["auth.py"],
        merged_at="2024-01-01T00:00:00Z",
    )
    dataset = Dataset(repo="a/b", examples=[example], dev_cutoff="2025-01-01T00:00:00Z")

    retriever = BM25Retriever()

    run_dir = harness.run_eval(retriever, dataset, "dev", "test-run", top_k=5)

    assert (run_dir / "config.json").exists()
    assert (run_dir / "predictions.jsonl").exists()
    assert "recall@5" in (run_dir / "metrics.json").read_text()


def test_run_eval_limit_runs_only_the_first_n_examples(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(metrics, "recall_at_k", lambda ranked, gold, k: 1.0)
    monkeypatch.setattr(metrics, "hit_at_k", lambda ranked, gold, k: 1.0)
    monkeypatch.setattr(
        harness, "index_repo_at_sha", lambda repo, sha: {"auth.py": "def login(): pass"}
    )

    examples = [
        IssueExample(
            repo="a/b",
            issue_number=i,
            issue_title="t",
            issue_body="b",
            base_sha="abc",
            fix_sha="def",
            gold_files=["auth.py"],
            merged_at=f"2024-01-{i:02d}T00:00:00Z",
        )
        for i in range(1, 6)
    ]
    dataset = Dataset(repo="a/b", examples=examples, dev_cutoff="2025-01-01T00:00:00Z")

    run_dir = harness.run_eval(BM25Retriever(), dataset, "dev", "test-run-limited", limit=2)

    assert json.loads((run_dir / "metrics.json").read_text())["n"] == 2
    assert json.loads((run_dir / "config.json").read_text())["limit"] == 2
