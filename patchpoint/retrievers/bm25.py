"""A from-scratch BM25 (Okapi) retriever — no external ranking dependency."""

from __future__ import annotations

import math
import re
from collections import Counter

from patchpoint.retrievers.base import Retriever
from patchpoint.schemas import ScoredFile

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever(Retriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._paths: list[str] = []
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._avg_doc_len = 0.0
        self._idf: dict[str, float] = {}

    def index(self, files: dict[str, str]) -> None:
        self._paths = list(files.keys())
        self._doc_freqs = []
        self._doc_lens = []
        df: Counter[str] = Counter()

        for path in self._paths:
            tokens = tokenize(files[path])
            self._doc_freqs.append(Counter(tokens))
            self._doc_lens.append(len(tokens))
            df.update(set(tokens))

        n_docs = len(self._paths)
        self._avg_doc_len = sum(self._doc_lens) / n_docs if n_docs else 0.0
        self._idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def query(self, issue_text: str, top_k: int = 10) -> list[ScoredFile]:
        query_terms = tokenize(issue_text)
        scores = [0.0] * len(self._paths)

        for i, doc_freqs in enumerate(self._doc_freqs):
            doc_len = self._doc_lens[i]
            for term in query_terms:
                freq = doc_freqs.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (
                    1 - self.b + self.b * doc_len / (self._avg_doc_len or 1)
                )
                scores[i] += idf * (freq * (self.k1 + 1)) / denom

        ranked = sorted(zip(self._paths, scores), key=lambda x: x[1], reverse=True)
        return [ScoredFile(path=p, score=s) for p, s in ranked[:top_k]]
