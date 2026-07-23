import pytest

from coalmine.judging.anchors import (
    build_synthetic_anchors,
    calibrate_judge,
    wilson_interval,
)
from coalmine.judging.judges import NoisyOracleJudge
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool


@pytest.fixture
def anchors():
    queries = synthetic_queries(60)
    pool = ResponsePool.build_synthetic(queries)
    return build_synthetic_anchors(queries, pool, n=300, seed=5)


def test_anchor_labels_are_balanced(anchors):
    a_labels = sum(item.label == "a" for item in anchors)
    assert 100 < a_labels < 200


def test_wilson_interval_sane():
    low, high = wilson_interval(85, 100)
    assert low < 0.85 < high
    assert 0.76 < low < 0.80
    assert 0.89 < high < 0.92
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


async def test_perfect_judge_calibrates_to_one(anchors):
    result = await calibrate_judge(NoisyOracleJudge(accuracy=1.0, tie_rate=0.0), anchors, seed=3)
    assert result.accuracy == 1.0
    assert result.n_decisive == 300


async def test_noisy_judge_calibrates_near_true_accuracy(anchors):
    result = await calibrate_judge(
        NoisyOracleJudge(accuracy=0.80, tie_rate=0.1), anchors, seed=3
    )
    assert result.wilson_low < 0.80 < result.wilson_high
    assert result.n_decisive < 300  # some ties excluded


async def test_degraded_judge_detectable(anchors):
    healthy = await calibrate_judge(NoisyOracleJudge(accuracy=0.85, tie_rate=0.1), anchors, seed=3)
    degraded = await calibrate_judge(NoisyOracleJudge(accuracy=0.70, tie_rate=0.1), anchors, seed=4)
    # The design-tolerance rule from the drift experiment: CI upper below 0.80.
    assert healthy.wilson_high > 0.80
    assert degraded.wilson_high < 0.80


async def test_calibration_reproducible(anchors):
    judge = NoisyOracleJudge(accuracy=0.85, tie_rate=0.1)
    r1 = await calibrate_judge(judge, anchors, seed=3, round_index=2)
    r2 = await calibrate_judge(judge, anchors, seed=3, round_index=2)
    assert r1 == r2
