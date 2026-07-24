import numpy as np
import pytest

from coalmine.sim.runner import run_stitched_batch
from coalmine.sim.verdicts import gen_wins
from coalmine.stats.confseq import StitchedCS, stitched_radius


def test_radius_shrinks_with_n():
    r = [float(stitched_radius(n, 0.05)) for n in (10, 100, 1_000, 10_000)]
    assert r[0] > r[1] > r[2] > r[3]
    assert r[3] < 0.025  # tight enough to be useful at 10k observations


def test_radius_grows_as_alpha_shrinks():
    assert float(stitched_radius(1000, 0.01)) > float(stitched_radius(1000, 0.10))


def test_batch_matches_scalar():
    rng = np.random.default_rng(41)
    wins = gen_wins(rng, trials=40, n=3000, win_rate=0.58)
    times = run_stitched_batch(wins, p0=0.5, alpha=0.05)
    for i in range(wins.shape[0]):
        cs = StitchedCS(p0=0.5, alpha=0.05)
        scalar_time = -1
        for j in range(wins.shape[1]):
            if cs.update(bool(wins[i, j])):
                scalar_time = j + 1
                break
        assert times[i] == scalar_time


def test_null_false_rejection_within_alpha_seeded():
    rng = np.random.default_rng(43)
    null = gen_wins(rng, trials=2000, n=10_000, win_rate=0.5)
    times = run_stitched_batch(null, p0=0.5, alpha=0.05)
    fpr = float((times > 0).mean())
    assert fpr < 0.05, f"time-uniform guarantee violated: {fpr}"


def test_detects_large_improvement():
    rng = np.random.default_rng(47)
    alt = gen_wins(rng, trials=800, n=10_000, win_rate=0.60)
    times = run_stitched_batch(alt, p0=0.5, alpha=0.05)
    assert float((times > 0).mean()) > 0.99
    assert float(np.median(times[times > 0])) < 1_500


def test_invalid_params():
    with pytest.raises(ValueError):
        StitchedCS(p0=1.5)
    with pytest.raises(ValueError):
        StitchedCS(p0=0.5, alpha=0.0)
