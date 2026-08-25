from patchpoint.retrievers.bm25 import BM25Retriever


def test_bm25_ranks_exact_term_match_first() -> None:
    retriever = BM25Retriever()
    retriever.index(
        {
            "a.py": "def login(username, password): pass",
            "b.py": "def generate_invoice(order): pass",
        }
    )
    ranked = retriever.query("login fails with password", top_k=2)
    assert ranked[0].path == "a.py"


def test_bm25_empty_index_returns_empty() -> None:
    retriever = BM25Retriever()
    retriever.index({})
    assert retriever.query("anything") == []
