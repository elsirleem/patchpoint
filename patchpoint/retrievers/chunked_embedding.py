"""Dense retriever that chunks each file before embedding it.

Scores every chunk against the issue text, then takes the max chunk score per
file as that file's score — file-level ranking is still the deliverable (see
CLAUDE.md's out-of-scope note on function-level localisation), chunking is only
about giving the embedding model a better shot at a file whose relevant part is
buried past its max sequence length.

Parameterized by a Chunker so the same retriever class serves both the fixed-
window and AST-chunking comparison — see chunking/fixed_window.py and
chunking/ast_chunk.py.
"""

from __future__ import annotations

import numpy as np

from patchpoint.chunking.base import Chunker
from patchpoint.retrievers.base import Retriever
from patchpoint.retrievers.embedding import cosine_scores, embed_many
from patchpoint.schemas import ScoredFile


class ChunkedEmbeddingRetriever(Retriever):
    def __init__(self, chunker: Chunker) -> None:
        self.chunker = chunker
        self._chunk_paths: list[str] = []
        self._embeddings: np.ndarray = np.zeros((0, 0))

    def index(self, files: dict[str, str]) -> None:
        chunks = [c for path, content in files.items() for c in self.chunker.chunk(path, content)]
        self._chunk_paths = [c.path for c in chunks]
        self._embeddings = (
            embed_many([c.text for c in chunks]) if chunks else np.zeros((0, 0))
        )

    def query(self, issue_text: str, top_k: int = 10) -> list[ScoredFile]:
        if not self._chunk_paths:
            return []
        query_vec = embed_many([issue_text])[0]
        chunk_scores = cosine_scores(self._embeddings, query_vec)

        file_scores: dict[str, float] = {}
        for path, score in zip(self._chunk_paths, chunk_scores.tolist(), strict=True):
            if score > file_scores.get(path, float("-inf")):
                file_scores[path] = score

        ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        return [ScoredFile(path=p, score=s) for p, s in ranked[:top_k]]
