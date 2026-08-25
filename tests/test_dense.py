from patchpoint.retrievers.dense import HashingDenseRetriever


def test_dense_is_deterministic_across_instances() -> None:
    files = {"a.py": "def login(username, password): pass", "b.py": "def invoice(order): pass"}
    r1 = HashingDenseRetriever()
    r1.index(files)
    r2 = HashingDenseRetriever()
    r2.index(files)
    assert r1.query("login password")[0].path == r2.query("login password")[0].path


def test_dense_prefers_lexically_similar_file() -> None:
    retriever = HashingDenseRetriever()
    retriever.index(
        {
            "a.py": "def login(username, password): authenticate user session",
            "b.py": "def generate_invoice(order): compute total billing price",
        }
    )
    ranked = retriever.query("user login authenticate session", top_k=2)
    assert ranked[0].path == "a.py"
