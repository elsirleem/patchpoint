from patchpoint.schemas import Dataset, IssueExample


def test_issue_example_round_trips_through_json() -> None:
    example = IssueExample(
        repo="pallets/flask",
        issue_number=1,
        issue_title="t",
        issue_body="b",
        base_sha="abc",
        fix_sha="def",
        gold_files=["a.py"],
        merged_at="2024-01-01T00:00:00Z",
    )
    dataset = Dataset(repo="pallets/flask", examples=[example], dev_cutoff="2024-01-01T00:00:00Z")
    restored = Dataset.model_validate_json(dataset.model_dump_json())
    assert restored.examples[0].gold_files == ["a.py"]
