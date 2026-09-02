import pytest

from patchpoint.eval.metrics import hit_at_k, recall_at_k
from patchpoint.schemas import ScoredFile


def _ranked(*paths: str) -> list[ScoredFile]:
    return [ScoredFile(path=p, score=1.0 - i * 0.01) for i, p in enumerate(paths)]


def test_recall_at_k_full_match() -> None:
    ranked = _ranked("a.py", "b.py", "c.py")
    assert recall_at_k(ranked, ["a.py", "b.py"], k=3) == 1.0


def test_recall_at_k_partial_match() -> None:
    ranked = _ranked("a.py", "x.py", "y.py")
    assert recall_at_k(ranked, ["a.py", "b.py"], k=3) == 0.5


def test_recall_at_k_no_match() -> None:
    ranked = _ranked("x.py", "y.py")
    assert recall_at_k(ranked, ["a.py", "b.py"], k=2) == 0.0


def test_recall_at_k_truncates_ranked_to_k_itself() -> None:
    # gold file is ranked 3rd — outside top 2 — even though the caller passed a
    # longer list than k. The function must not trust ranked was pre-truncated.
    ranked = _ranked("x.py", "y.py", "a.py")
    assert recall_at_k(ranked, ["a.py"], k=2) == 0.0
    assert recall_at_k(ranked, ["a.py"], k=3) == 1.0


def test_recall_at_k_raises_on_empty_gold_files() -> None:
    with pytest.raises(ValueError):
        recall_at_k(_ranked("a.py"), [], k=5)


def test_hit_at_k_any_match_is_one() -> None:
    ranked = _ranked("x.py", "a.py")
    assert hit_at_k(ranked, ["a.py", "b.py"], k=2) == 1.0


def test_hit_at_k_no_match_is_zero() -> None:
    ranked = _ranked("x.py", "y.py")
    assert hit_at_k(ranked, ["a.py"], k=2) == 0.0


def test_hit_at_k_truncates_ranked_to_k_itself() -> None:
    ranked = _ranked("x.py", "y.py", "a.py")
    assert hit_at_k(ranked, ["a.py"], k=2) == 0.0
    assert hit_at_k(ranked, ["a.py"], k=3) == 1.0


def test_hit_at_k_raises_on_empty_gold_files() -> None:
    with pytest.raises(ValueError):
        hit_at_k(_ranked("a.py"), [], k=5)
