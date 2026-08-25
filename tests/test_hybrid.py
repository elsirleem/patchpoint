import pytest

from patchpoint.retrievers.bm25 import BM25Retriever
from patchpoint.retrievers.dense import HashingDenseRetriever
from patchpoint.retrievers.hybrid import RRFHybridRetriever


def test_hybrid_fuses_both_retrievers() -> None:
    files = {
        "a.py": "def login(username, password): authenticate user session",
        "b.py": "def generate_invoice(order): compute total billing price",
    }
    retriever = RRFHybridRetriever([BM25Retriever(), HashingDenseRetriever()])
    retriever.index(files)
    ranked = retriever.query("user login authenticate", top_k=2)
    assert ranked[0].path == "a.py"


def test_hybrid_requires_at_least_one_retriever() -> None:
    with pytest.raises(ValueError):
        RRFHybridRetriever([])
