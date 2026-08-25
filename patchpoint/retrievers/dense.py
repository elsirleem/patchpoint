"""Offline dense retriever: a hashing-trick embedding stand-in for a real model.

Deterministic and dependency-free beyond numpy, so `make demo` never needs network
access or a downloaded model. Swap for real embeddings once evaluated against BM25.
Uses hashlib rather than Python's builtin hash() because the latter is randomized
per-process (PYTHONHASHSEED) and would break reproducibility across runs.
"""

from __future__ import annotations

import hashlib

import numpy as np

from patchpoint.retrievers.base import Retriever
from patchpoint.retrievers.bm25 import tokenize
from patchpoint.schemas import ScoredFile

_DIM = 256


def _stable_hash(token: str) -> int:
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)


def _hash_embed(text: str, dim: int = _DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    for token in tokenize(text):
        h = _stable_hash(token)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class HashingDenseRetriever(Retriever):
    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim
        self._paths: list[str] = []
        self._embeddings: np.ndarray = np.zeros((0, dim))

    def index(self, files: dict[str, str]) -> None:
        self._paths = list(files.keys())
        self._embeddings = (
            np.stack([_hash_embed(files[p], self.dim) for p in self._paths])
            if self._paths
            else np.zeros((0, self.dim))
        )

    def query(self, issue_text: str, top_k: int = 10) -> list[ScoredFile]:
        if not self._paths:
            return []
        query_vec = _hash_embed(issue_text, self.dim)
        scores = self._embeddings @ query_vec
        ranked = sorted(zip(self._paths, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [ScoredFile(path=p, score=s) for p, s in ranked[:top_k]]
