from patchpoint.data import dataset as dataset_module
from patchpoint.schemas import IssueExample


def _example(n: int, merged_at: str) -> IssueExample:
    return IssueExample(
        repo="a/b",
        issue_number=n,
        issue_title="t",
        issue_body="b",
        base_sha="abc",
        fix_sha="def",
        gold_files=["a.py"],
        merged_at=merged_at,
    )


def test_build_dataset_splits_chronologically(monkeypatch) -> None:
    examples = [_example(i, f"2024-01-{i:02d}T00:00:00Z") for i in range(1, 11)]
    monkeypatch.setattr(dataset_module, "mine_examples", lambda repo, max_examples: examples)

    dataset = dataset_module.build_dataset("a/b", max_examples=10, dev_fraction=0.8)

    dev = [e for e in dataset.examples if e.merged_at < dataset.dev_cutoff]
    test = [e for e in dataset.examples if e.merged_at >= dataset.dev_cutoff]
    assert len(dev) == 7
    assert len(test) == 3
    assert max(e.merged_at for e in dev) < min(e.merged_at for e in test)
