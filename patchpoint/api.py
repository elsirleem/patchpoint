"""Minimal FastAPI service exposing retrieval over a pre-built dataset/index."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="patchpoint")


class RankRequest(BaseModel):
    repo: str
    issue_text: str
    top_k: int = 10


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rank")
def rank(request: RankRequest) -> None:
    # Needs the `index` CLI command wired up first — see CLAUDE.md status.
    raise HTTPException(status_code=501, detail="indexing is not wired in yet")
