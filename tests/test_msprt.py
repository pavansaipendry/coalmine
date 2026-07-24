import math

import numpy as np
import pytest

from coalmine.sim.runner import run_msprt_batch
from coalmine.sim.verdicts import gen_wins
from coalmine.stats.msprt import MixtureSPRT


def test_log_lambda_matches_hand_computation():
    # p0 = 0.5, uniform prior truncated to (0.5, 1), one win:
    # Lambda = [int_{.5}^{1} p dp / 0.5] / 0.5 = 0.75 / 0.5 / 0.5 = 1.5
    t = MixtureSPRT(p0=0.5, alpha=0.05)
    t.update(True)
    assert t.log_lambda == pytest.approx(math.log(1.5), abs=1e-9)


def test_rejection_is_sticky_and_reset_clears():
    t = MixtureSPRT(p0=0.5, alpha=0.05)
    while not t.update(True):
        pass
    n_at_rejection = t.n
    t.update(False)
    assert t.n == n_at_rejection
    t.reset()
    assert not t.rejected and t.n == 0


def test_batch_matches_scalar():
    rng = np.random.default_rng(23)
    wins = gen_wins(rng, trials=40, n=2000, win_rate=0.55)
    times = run_msprt_batch(wins, p0=0.5, alpha=0.05)
    for i in range(wins.shape[0]):
        t = MixtureSPRT(p0=0.5, alpha=0.05)
        scalar_time = -1
        for j in range(wins.shape[1]):
            if t.update(bool(wins[i, j])):
                scalar_time = j + 1
                break
        assert times[i] == scalar_time, f"trial {i}: batch {times[i]} != scalar {scalar_time}"


def test_null_false_rejection_within_alpha_seeded():
    rng = np.random.default_rng(29)
    null = gen_wins(rng, trials=2000, n=10_000, win_rate=0.5)
    times = run_msprt_batch(null, p0=0.5, alpha=0.05)
    fpr = float((times > 0).mean())
    assert fpr <= 0.06, f"anytime guarantee violated: {fpr}"


def test_detects_true_improvement():
    rng = np.random.default_rng(31)
    alt = gen_wins(rng, trials=1000, n=10_000, win_rate=0.56)
    times = run_msprt_batch(alt, p0=0.5, alpha=0.05)
    assert float((times > 0).mean()) > 0.95
    assert float(np.median(times[times > 0])) < 2_000


def test_one_sided_ignores_regressions():
    rng = np.random.default_rng(37)
    worse = gen_wins(rng, trials=1000, n=5_000, win_rate=0.44)
    times = run_msprt_batch(worse, p0=0.5, alpha=0.05)
    assert float((times > 0).mean()) < 0.01


def test_invalid_params():
    with pytest.raises(ValueError):
        MixtureSPRT(p0=0.0)
    with pytest.raises(ValueError):
        MixtureSPRT(p0=0.5, alpha=1.0)
    with pytest.raises(ValueError):
        MixtureSPRT(p0=0.5, prior_a=0.0)
