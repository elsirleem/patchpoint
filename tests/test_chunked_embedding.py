import numpy as np

from patchpoint.chunking.fixed_window import FixedWindowChunker
from patchpoint.retrievers import embedding
from patchpoint.retrievers.chunked_embedding import ChunkedEmbeddingRetriever


class _FakeModel:
    """Embeds by whether the text mentions 'target' — deterministic, no download."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return np.array([[1.0, 0.0] if "target" in t else [0.0, 1.0] for t in texts])


def test_chunked_retriever_finds_relevant_chunk_in_an_otherwise_irrelevant_file(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(embedding, "_get_model", lambda: _FakeModel())

    # a big file where only one small chunk is actually relevant
    filler = "\n".join(f"noise line {i}" for i in range(60))
    content = f"{filler}\ndef target_function():\n    pass\n{filler}"
    other = "\n".join(f"unrelated line {i}" for i in range(10))

    retriever = ChunkedEmbeddingRetriever(FixedWindowChunker(window_lines=20, overlap_lines=0))
    retriever.index({"big.py": content, "other.py": other})
    ranked = retriever.query("target issue", top_k=2)

    assert ranked[0].path == "big.py"


def test_chunked_retriever_empty_index_returns_empty() -> None:
    retriever = ChunkedEmbeddingRetriever(FixedWindowChunker())
    retriever.index({})
    assert retriever.query("anything") == []


def test_chunked_retriever_uses_max_score_across_a_files_chunks(monkeypatch) -> None:
    monkeypatch.setattr(embedding, "_get_model", lambda: _FakeModel())

    retriever = ChunkedEmbeddingRetriever(FixedWindowChunker(window_lines=1, overlap_lines=0))
    # 3 chunks for a.py, only the middle one matches "target"
    retriever.index({"a.py": "irrelevant\ntarget stuff\nalso irrelevant"})
    ranked = retriever.query("target", top_k=1)

    assert ranked[0].path == "a.py"
    assert ranked[0].score == 1.0  # the max, not an average pulled down by the other chunks
