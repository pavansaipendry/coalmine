import numpy as np
import pytest

from coalmine.judging.ensemble import (
    EnsembleJudge,
    calibrate_disagreement_threshold,
    rolling_disagreement,
)
from coalmine.judging.judges import NoisyOracleJudge, compare_randomized

GOOD = {"text": "grounded", "source_pool": "good"}
BAD = {"text": "vague", "source_pool": "bad"}


def _members(accuracies, tie_rate=0.1):
    return [
        NoisyOracleJudge(accuracy=a, tie_rate=tie_rate, judge_id=f"m{i}")
        for i, a in enumerate(accuracies)
    ]


def _rng(i):
    return np.random.default_rng([77, i])


async def test_majority_vote_beats_single_member():
    ensemble = EnsembleJudge(_members([0.8, 0.8, 0.8], tie_rate=0.0))
    correct = 0
    n = 3_000
    for i in range(n):
        result = await compare_randomized(ensemble, "q", GOOD, BAD, _rng(i))
        correct += result.winner == "a"
    # Majority of three 0.8 judges: 0.8^3 + 3*0.8^2*0.2 = 0.896
    assert correct / n > 0.86


async def test_majority_absorbs_one_degraded_member():
    healthy = EnsembleJudge(_members([0.85, 0.85, 0.85], tie_rate=0.0))
    degraded = EnsembleJudge(_members([0.85, 0.85, 0.55], tie_rate=0.0))
    h_correct = d_correct = 0
    n = 3_000
    for i in range(n):
        h_correct += (await compare_randomized(healthy, "q", GOOD, BAD, _rng(i))).winner == "a"
        d_correct += (await compare_randomized(degraded, "q", GOOD, BAD, _rng(i))).winner == "a"
    assert d_correct / n > 0.78, "vote should hold up despite one bad member"
    assert h_correct / n - d_correct / n < 0.10


async def test_disagreement_rate_rises_with_degradation():
    healthy = EnsembleJudge(_members([0.85, 0.85, 0.85]))
    degraded = EnsembleJudge(_members([0.85, 0.85, 0.55]))
    n = 2_000
    for i in range(n):
        await compare_randomized(healthy, "q", GOOD, BAD, _rng(i))
        await compare_randomized(degraded, "q", GOOD, BAD, _rng(i))
    healthy_rate = float(np.mean(healthy.disagreements))
    degraded_rate = float(np.mean(degraded.disagreements))
    assert degraded_rate > healthy_rate + 0.05


async def test_deterministic_per_rng():
    ensemble_a = EnsembleJudge(_members([0.85, 0.85, 0.85]))
    ensemble_b = EnsembleJudge(_members([0.85, 0.85, 0.85]))
    for i in range(100):
        ra = await ensemble_a.compare_detailed("q", GOOD, BAD, _rng(i))
        rb = await ensemble_b.compare_detailed("q", GOOD, BAD, _rng(i))
        assert ra == rb


def test_rolling_and_threshold():
    stream = [False] * 500 + [True] * 500
    rolled = rolling_disagreement(stream, window=100)
    assert rolled[0] == 0.0 and rolled[-1] == pytest.approx(1.0)
    thr = calibrate_disagreement_threshold(0.3, window=300, n_checks=2_000, false_alarm_rate=0.05)
    assert 0.3 < thr < 0.45


def test_needs_two_members():
    with pytest.raises(ValueError):
        EnsembleJudge(_members([0.85]))
