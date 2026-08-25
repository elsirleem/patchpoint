"""Pydantic models shared across module boundaries."""

from __future__ import annotations

from pydantic import BaseModel


class ScoredFile(BaseModel):
    """A single file ranked by a retriever."""

    path: str
    score: float


class IssueExample(BaseModel):
    """One (issue, fix) pair mined from git history."""

    repo: str
    issue_number: int
    issue_title: str
    issue_body: str
    base_sha: str
    """Parent commit of the fix — the state a retriever indexes and is scored against."""
    fix_sha: str
    gold_files: list[str]
    """Files touched by the fix commit(s). Never exposed to a retriever."""
    merged_at: str
    """ISO 8601 timestamp, used for the chronological dev/test split."""


class Dataset(BaseModel):
    """A built dataset: examples plus the chronological split boundary."""

    repo: str
    examples: list[IssueExample]
    dev_cutoff: str
    """ISO 8601 timestamp. Examples with merged_at < this go to dev, the rest to test."""


class RunConfig(BaseModel):
    """Everything needed to reproduce a single eval run."""

    run_id: str
    repo: str
    base_sha: str
    retriever: str
    split: str
    top_k: int = 10


class Prediction(BaseModel):
    """One retriever's output for one issue, as written to predictions.jsonl."""

    issue_number: int
    ranked_files: list[ScoredFile]
