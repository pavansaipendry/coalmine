import numpy as np
import pytest

from coalmine.sim.runner import run_sprt_batch
from coalmine.sim.verdicts import gen_wins
from coalmine.stats.sprt import SPRT, Decision


def test_all_wins_accepts_h1_quickly():
    t = SPRT(p0=0.5, p1=0.55, alpha=0.05, beta=0.10)
    while t.update(True) is Decision.CONTINUE:
        pass
    assert t.decision is Decision.ACCEPT_H1
    # upper / log(p1/p0) = 2.89 / 0.0953 -> 31 straight wins
    assert t.n == 31


def test_all_losses_accepts_h0():
    t = SPRT(p0=0.5, p1=0.55, alpha=0.05, beta=0.10)
    while t.update(False) is Decision.CONTINUE:
        pass
    assert t.decision is Decision.ACCEPT_H0


def test_decision_is_sticky_and_reset_clears():
    t = SPRT(p0=0.5, p1=0.55)
    while t.update(True) is Decision.CONTINUE:
        pass
    n_at_decision = t.n
    t.update(False)
    assert t.n == n_at_decision  # no updates after a decision
    t.reset()
    assert t.decision is Decision.CONTINUE and t.n == 0 and t.llr == 0.0


def test_error_rates_within_tolerance_seeded():
    # Wald guarantees alpha/beta up to overshoot; with a fixed seed this is deterministic.
    rng = np.random.default_rng(7)
    null = gen_wins(rng, trials=2000, n=8000, win_rate=0.5)
    dec_null, _ = run_sprt_batch(null, p0=0.5, p1=0.55, alpha=0.05, beta=0.10)
    fpr = float((dec_null == 1).mean())
    assert fpr < 0.06, f"false promotion rate {fpr} exceeds alpha with margin"

    alt = gen_wins(rng, trials=2000, n=8000, win_rate=0.55)
    dec_alt, _ = run_sprt_batch(alt, p0=0.5, p1=0.55, alpha=0.05, beta=0.10)
    miss = float((dec_alt == -1).mean())
    assert miss < 0.11, f"miss rate {miss} exceeds beta with margin"


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        SPRT(p0=0.5, p1=0.5)
    with pytest.raises(ValueError):
        SPRT(p0=0.5, p1=0.55, alpha=0.0)
