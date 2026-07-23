import pytest

from coalmine.stats.channel import JudgeChannel


def test_round_trip_inversion():
    ch = JudgeChannel(accuracy=0.85, tie_rate=0.10)
    for w in [0.30, 0.45, 0.50, 0.55, 0.70]:
        assert ch.true_win_rate(ch.observed_win_rate(w)) == pytest.approx(w)


def test_attenuation_example():
    # The worked example from the design: a=0.85 shrinks a 5-point effect to 3.5 points.
    ch = JudgeChannel(accuracy=0.85, tie_rate=0.10)
    assert ch.observed_win_rate(0.55) == pytest.approx(0.535)
    assert ch.attenuation() == pytest.approx(0.70)


def test_fair_coin_is_fixed_point():
    ch = JudgeChannel(accuracy=0.80, tie_rate=0.20)
    assert ch.observed_win_rate(0.5) == pytest.approx(0.5)


def test_requests_per_verdict():
    ch = JudgeChannel(accuracy=0.85, tie_rate=0.10)
    assert ch.requests_per_verdict(0.20) == pytest.approx(1.0 / (0.20 * 0.90))


def test_rejects_uninformative_judge():
    with pytest.raises(ValueError):
        JudgeChannel(accuracy=0.5)
    with pytest.raises(ValueError):
        JudgeChannel(accuracy=0.85, tie_rate=1.0)
