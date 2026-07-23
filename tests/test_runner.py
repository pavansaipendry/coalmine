"""Cross-validation: the vectorized Monte Carlo harness must agree exactly,
trial for trial, with the scalar SPRT/CUSUM implementations on identical streams."""

import numpy as np

from coalmine.sim.runner import run_cusum_batch, run_sprt_batch
from coalmine.sim.verdicts import gen_wins
from coalmine.stats.cusum import CUSUM
from coalmine.stats.sprt import SPRT, Decision


def _scalar_sprt(row, p0, p1, alpha, beta):
    t = SPRT(p0=p0, p1=p1, alpha=alpha, beta=beta)
    for win in row:
        if t.update(bool(win)) is not Decision.CONTINUE:
            break
    code = {Decision.ACCEPT_H1: 1, Decision.ACCEPT_H0: -1, Decision.CONTINUE: 0}[t.decision]
    return code, t.n if code != 0 else len(row)


def _scalar_cusum(row, p0, p1, h):
    c = CUSUM(p0=p0, p1=p1, h=h)
    for win in row:
        if c.update(bool(win)):
            return c.n
    return -1


def test_sprt_batch_matches_scalar():
    rng = np.random.default_rng(3)
    wins = gen_wins(rng, trials=50, n=3000, win_rate=0.52)
    dec, stops = run_sprt_batch(wins, p0=0.5, p1=0.55, alpha=0.05, beta=0.10)
    for i in range(wins.shape[0]):
        exp_dec, exp_stop = _scalar_sprt(wins[i], 0.5, 0.55, 0.05, 0.10)
        assert dec[i] == exp_dec, f"trial {i}: decision {dec[i]} != scalar {exp_dec}"
        assert stops[i] == exp_stop, f"trial {i}: stop {stops[i]} != scalar {exp_stop}"


def test_cusum_batch_matches_scalar():
    rng = np.random.default_rng(5)
    wins = gen_wins(rng, trials=50, n=2000, win_rate=0.47)
    alarms = run_cusum_batch(wins, p0=0.5, p1=0.45, h=4.0)
    for i in range(wins.shape[0]):
        expected = _scalar_cusum(wins[i], 0.5, 0.45, 4.0)
        assert alarms[i] == expected, f"trial {i}: alarm {alarms[i]} != scalar {expected}"


def test_sprt_batch_censoring():
    # A stream too short to decide must come back censored with stop == n.
    wins = np.tile(np.array([[True, False]]), (1, 5))  # 10 alternating verdicts
    dec, stops = run_sprt_batch(wins, p0=0.5, p1=0.55)
    assert dec[0] == 0
    assert stops[0] == wins.shape[1]
