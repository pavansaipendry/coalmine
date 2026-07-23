import numpy as np
import pytest

from coalmine.judging.judges import NoisyOracleJudge, compare_randomized

GOOD = {"text": "grounded answer", "source_pool": "good"}
BAD = {"text": "vague answer", "source_pool": "bad"}


def _rng(i: int) -> np.random.Generator:
    return np.random.default_rng([99, i])


async def test_perfect_judge_always_picks_good():
    judge = NoisyOracleJudge(accuracy=1.0, tie_rate=0.0)
    for i in range(200):
        result = await compare_randomized(judge, "q", GOOD, BAD, _rng(i))
        assert result.winner == "a"
        result = await compare_randomized(judge, "q", BAD, GOOD, _rng(i))
        assert result.winner == "b"


async def test_randomization_maps_verdicts_back_correctly():
    # Whatever the presentation order, a perfect judge's config-space verdict
    # must name the good config — this is the swap-mapping correctness check.
    judge = NoisyOracleJudge(accuracy=1.0, tie_rate=0.0)
    first_shown = set()
    for i in range(100):
        result = await compare_randomized(judge, "q", GOOD, BAD, _rng(i))
        first_shown.add(result.first_shown)
        assert result.winner == "a"
    assert first_shown == {"a", "b"}, "randomization should present both orders"


async def test_equal_quality_splits_evenly():
    judge = NoisyOracleJudge(accuracy=0.85, tie_rate=0.0)
    wins_a = 0
    for i in range(3000):
        result = await compare_randomized(judge, "q", GOOD, GOOD, _rng(i))
        wins_a += result.winner == "a"
    assert abs(wins_a / 3000 - 0.5) < 0.03


async def test_tie_rate_respected():
    judge = NoisyOracleJudge(accuracy=0.85, tie_rate=0.25)
    ties = 0
    for i in range(2000):
        result = await compare_randomized(judge, "q", GOOD, BAD, _rng(i))
        ties += result.winner == "tie"
    assert abs(ties / 2000 - 0.25) < 0.03


async def test_accuracy_observed_on_unequal_pairs():
    judge = NoisyOracleJudge(accuracy=0.85, tie_rate=0.0)
    correct = 0
    for i in range(4000):
        result = await compare_randomized(judge, "q", GOOD, BAD, _rng(i))
        correct += result.winner == "a"
    assert abs(correct / 4000 - 0.85) < 0.02


async def test_position_bias_raises_first_win_rate_on_equal_pairs():
    biased = NoisyOracleJudge(accuracy=0.85, tie_rate=0.0, position_bias=0.3)
    first_wins = 0
    for i in range(4000):
        result = await compare_randomized(biased, "q", GOOD, GOOD, _rng(i))
        first_wins += result.winner == result.first_shown
    # P(first wins | equal) = bias + (1 - bias)/2 = 0.65
    assert abs(first_wins / 4000 - 0.65) < 0.02


async def test_comparison_is_reproducible():
    judge = NoisyOracleJudge(accuracy=0.85, tie_rate=0.1, position_bias=0.1)
    for i in range(50):
        r1 = await compare_randomized(judge, "q", GOOD, BAD, _rng(i))
        r2 = await compare_randomized(judge, "q", GOOD, BAD, _rng(i))
        assert r1 == r2


async def test_oracle_requires_rng():
    judge = NoisyOracleJudge()
    with pytest.raises(ValueError):
        await judge.compare("q", GOOD, BAD, rng=None)


def test_invalid_params():
    with pytest.raises(ValueError):
        NoisyOracleJudge(accuracy=0.5)
    with pytest.raises(ValueError):
        NoisyOracleJudge(tie_rate=1.0)
    with pytest.raises(ValueError):
        NoisyOracleJudge(position_bias=-0.1)
