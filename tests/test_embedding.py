import numpy as np

from patchpoint.retrievers import embedding


class _FakeModel:
    """Deterministic stand-in for SentenceTransformer: embeds by first-letter."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return np.array([[1.0, 0.0] if t.startswith("a") else [0.0, 1.0] for t in texts])


def test_embedding_retriever_ranks_by_cosine_similarity(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(embedding, "_get_model", lambda: _FakeModel())

    retriever = embedding.SentenceEmbeddingRetriever()
    retriever.index({"a.py": "alpha content", "b.py": "beta content"})
    ranked = retriever.query("alpha issue", top_k=2)

    assert ranked[0].path == "a.py"


def test_embedding_retriever_caches_across_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls = []

    class _CountingModel(_FakeModel):
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            calls.append(list(texts))
            return super().encode(texts)

    monkeypatch.setattr(embedding, "_get_model", lambda: _CountingModel())

    retriever = embedding.SentenceEmbeddingRetriever()
    retriever.index({"a.py": "alpha content"})
    retriever.index({"a.py": "alpha content"})  # same content — should hit the cache

    assert len(calls) == 1


def test_embedding_retriever_empty_index_returns_empty() -> None:
    retriever = embedding.SentenceEmbeddingRetriever()
    retriever.index({})
    assert retriever.query("anything") == []
