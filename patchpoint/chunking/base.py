"""Chunk interface — how a file's content is split for indexing."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class Chunk(BaseModel):
    path: str
    start_line: int
    end_line: int
    text: str


class Chunker(Protocol):
    def chunk(self, path: str, content: str) -> list[Chunk]: ...
