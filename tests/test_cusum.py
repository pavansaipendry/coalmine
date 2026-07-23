import numpy as np
import pytest

from coalmine.sim.runner import calibrate_cusum_threshold, run_cusum_batch
from coalmine.sim.verdicts import gen_wins, gen_wins_changepoint
from coalmine.stats.cusum import CUSUM


def test_reflection_at_zero():
    c = CUSUM(p0=0.5, p1=0.45, h=3.0)
    for _ in range(50):
        c.update(True)  # wins push the drop-detector down; reflection holds it at 0
    assert c.stat == 0.0
    assert not c.alarmed


def test_alarm_fires_and_sticks_under_collapse():
    c = CUSUM(p0=0.5, p1=0.45, h=3.0)
    n = 0
    while not c.update(False):
        n += 1
    assert c.alarmed
    # h / |log((1-p1)/(1-p0))| = 3.0 / 0.0953 -> ~32 straight losses
    assert 25 <= c.n <= 40


def test_calibrated_threshold_hits_target_fpr_seeded():
    rng = np.random.default_rng(11)
    null = gen_wins(rng, trials=2000, n=5000, win_rate=0.5)
    h = calibrate_cusum_threshold(null, p0=0.5, p1=0.45, false_alarm_rate=0.05)
    fresh_null = gen_wins(rng, trials=2000, n=5000, win_rate=0.5)
    alarms = run_cusum_batch(fresh_null, p0=0.5, p1=0.45, h=h)
    fpr = float((alarms > 0).mean())
    assert 0.02 < fpr < 0.09, f"calibrated FPR {fpr} far from 5% target"


def test_detects_injected_regression_seeded():
    rng = np.random.default_rng(13)
    null = gen_wins(rng, trials=1000, n=6000, win_rate=0.5)
    h = calibrate_cusum_threshold(null, p0=0.5, p1=0.45, false_alarm_rate=0.05)
    wins = gen_wins_changepoint(
        rng, trials=500, n=6000, win_rate_before=0.5, win_rate_after=0.42, changepoint=2000
    )
    alarms = run_cusum_batch(wins, p0=0.5, p1=0.45, h=h)
    post = alarms[alarms > 2000]
    assert len(post) > 450  # a big shift should almost always be caught
    assert float(np.median(post - 2000)) < 600


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        CUSUM(p0=0.5, p1=0.55, h=3.0)  # wrong direction
    with pytest.raises(ValueError):
        CUSUM(p0=0.5, p1=0.45, h=0.0)
