"""Dense retriever using real sentence embeddings.

Replaces the hashing stand-in in retrievers/dense.py — see that module's own
docstring: "Swap for real embeddings once evaluated against BM25." sentence-
transformers is an optional dependency (the `embeddings` extra) so `make demo`
stays offline and dependency-free; the model is imported lazily, only when this
retriever is actually used.

Files are embedded whole, not chunked — chunking isn't wired in yet (see
chunking/), so a file longer than the model's max sequence length is silently
truncated by sentence-transformers itself. That's a known limitation, not an
oversight; revisit once chunking is integrated.

Embeddings are cached to disk per (model, text), keyed by content hash. A local
model has no API cost, but still costs real compute time across dozens of eval
re-runs — this follows CLAUDE.md's external-call caching rule anyway.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

from patchpoint.retrievers.base import Retriever
from patchpoint.schemas import ScoredFile

_MODEL_NAME = "all-MiniLM-L6-v2"
_CACHE_DIR = Path(".cache") / "embeddings" / _MODEL_NAME

_model = None


def _get_model():  # type: ignore[no-untyped-def]
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _cache_path(text: str) -> Path:
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{key}.json"


def _embed_many(texts: list[str]) -> np.ndarray:
    """Embeds `texts`, reusing cached vectors and batch-encoding the rest."""
    vectors_by_index: dict[int, list[float]] = {}
    to_encode: list[str] = []
    to_encode_idx: list[int] = []

    for i, text in enumerate(texts):
        path = _cache_path(text)
        if path.exists():
            vectors_by_index[i] = json.loads(path.read_text())
        else:
            to_encode.append(text)
            to_encode_idx.append(i)

    if to_encode:
        model = _get_model()
        encoded = model.encode(to_encode, normalize_embeddings=True, show_progress_bar=False)
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for i, text, vec in zip(to_encode_idx, to_encode, encoded, strict=True):
            vectors_by_index[i] = vec.tolist()
            _cache_path(text).write_text(json.dumps(vectors_by_index[i]))

    return np.array([vectors_by_index[i] for i in range(len(texts))])


class SentenceEmbeddingRetriever(Retriever):
    def __init__(self) -> None:
        self._paths: list[str] = []
        self._embeddings: np.ndarray = np.zeros((0, 0))

    def index(self, files: dict[str, str]) -> None:
        self._paths = list(files.keys())
        self._embeddings = (
            _embed_many([files[p] for p in self._paths]) if self._paths else np.zeros((0, 0))
        )

    def query(self, issue_text: str, top_k: int = 10) -> list[ScoredFile]:
        if not self._paths:
            return []
        query_vec = _embed_many([issue_text])[0]
        # OpenBLAS spuriously raises divide-by-zero/overflow/invalid-value FPE
        # warnings on some matrix shapes here even though inputs are unit-norm
        # and the output has no NaN/Inf — verified directly, not assumed. Known
        # OpenBLAS quirk, not a data bug; suppressed rather than left as noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scores = self._embeddings @ query_vec
        ranked = sorted(zip(self._paths, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [ScoredFile(path=p, score=s) for p, s in ranked[:top_k]]
