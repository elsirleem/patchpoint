from patchpoint.eval import harness, metrics
from patchpoint.retrievers.bm25 import BM25Retriever
from patchpoint.schemas import Dataset, IssueExample


def test_run_eval_writes_config_predictions_and_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(metrics, "recall_at_k", lambda ranked, gold, k: 1.0)
    monkeypatch.setattr(metrics, "hit_at_k", lambda ranked, gold, k: 1.0)

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
    retriever.index({"auth.py": "def login(): pass"})

    run_dir = harness.run_eval(retriever, dataset, "dev", "test-run", top_k=5)

    assert (run_dir / "config.json").exists()
    assert (run_dir / "predictions.jsonl").exists()
    assert "recall@5" in (run_dir / "metrics.json").read_text()
