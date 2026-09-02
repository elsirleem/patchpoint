import pytest

from patchpoint.eval.bootstrap import paired_bootstrap


def test_paired_bootstrap_identical_scores_gives_zero_diff_and_p_one() -> None:
    scores = [0.5, 0.6, 0.7, 0.8, 0.9]
    result = paired_bootstrap(scores, scores, n_resamples=1000)
    assert result["diff"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["p_value"] == 1.0


def test_paired_bootstrap_consistent_winner_has_significant_positive_diff() -> None:
    scores_a = [1.0] * 20
    scores_b = [0.0] * 20
    result = paired_bootstrap(scores_a, scores_b, n_resamples=2000)
    assert result["diff"] == 1.0
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.05


def test_paired_bootstrap_is_symmetric_under_swap() -> None:
    scores_a = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    scores_b = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ab = paired_bootstrap(scores_a, scores_b, n_resamples=2000)
    ba = paired_bootstrap(scores_b, scores_a, n_resamples=2000)
    assert ab["diff"] == pytest.approx(-ba["diff"])
    assert ab["p_value"] == pytest.approx(ba["p_value"])


def test_paired_bootstrap_is_deterministic_given_seed() -> None:
    scores_a = [1.0, 0.0, 1.0, 1.0, 0.0]
    scores_b = [0.0, 0.0, 1.0, 0.0, 0.0]
    r1 = paired_bootstrap(scores_a, scores_b, n_resamples=500, seed=42)
    r2 = paired_bootstrap(scores_a, scores_b, n_resamples=500, seed=42)
    assert r1 == r2


def test_paired_bootstrap_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 0.0], [1.0], n_resamples=10)


def test_paired_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([], [], n_resamples=10)
