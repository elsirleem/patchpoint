"""Minimal FastAPI service exposing retrieval over a live-indexed repo.

Indexes at the repo's current HEAD, not a historical base_sha: invariant 1 (never
index at HEAD) is about not leaking a fix into the eval corpus, which doesn't
apply here — a live issue has no fix commit yet, so HEAD is the only sensible
state to search. Uses BM25 since it's the only retriever validated as a real
baseline so far (see results/bm25-dev-pallets__flask/metrics.json).
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from patchpoint.data.checkout import index_repo_at_sha, resolve_head_sha
from patchpoint.retrievers.bm25 import BM25Retriever
from patchpoint.schemas import ScoredFile

app = FastAPI(title="patchpoint")


class RankRequest(BaseModel):
    repo: str
    issue_text: str
    top_k: int = 10


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rank")
def rank(request: RankRequest) -> list[ScoredFile]:
    sha = resolve_head_sha(request.repo)
    files = index_repo_at_sha(request.repo, sha)
    retriever = BM25Retriever()
    retriever.index(files)
    return retriever.query(request.issue_text, top_k=request.top_k)
