import numpy as np
import pytest

from coalmine.sim.runner import run_sprt_batch
from coalmine.sim.verdicts import gen_wins
from coalmine.stats.channel import JudgeChannel
from coalmine.stats.wald import accept_h1_probability, expected_n, plan


def test_boundary_probabilities_match_design_error_rates():
    # At p = p0 the accept-H1 probability is ~alpha; at p = p1 it is ~1 - beta.
    assert accept_h1_probability(0.5, 0.5, 0.55, 0.05, 0.10) == pytest.approx(0.05, abs=0.005)
    assert accept_h1_probability(0.55, 0.5, 0.55, 0.05, 0.10) == pytest.approx(0.90, abs=0.005)


def test_accept_probability_monotone_in_p():
    ps = np.linspace(0.40, 0.62, 12)
    probs = [accept_h1_probability(p, 0.5, 0.55, 0.05, 0.10) for p in ps]
    assert all(b >= a for a, b in zip(probs, probs[1:]))


def test_expected_n_peaks_between_hypotheses():
    inside = expected_n(0.525, 0.5, 0.55, 0.05, 0.10)
    assert inside > expected_n(0.50, 0.5, 0.55, 0.05, 0.10)
    assert inside > expected_n(0.55, 0.5, 0.55, 0.05, 0.10)
    assert inside > expected_n(0.40, 0.5, 0.55, 0.05, 0.10)


def test_theory_matches_simulation_seeded():
    # Wald's E[N] ignores overshoot so simulation runs slightly above it.
    rng = np.random.default_rng(17)
    wins = gen_wins(rng, trials=3000, n=6000, win_rate=0.55)
    _, stops = run_sprt_batch(wins, p0=0.5, p1=0.55, alpha=0.05, beta=0.10)
    empirical = float(stops.mean())
    theory = expected_n(0.55, 0.5, 0.55, 0.05, 0.10)
    assert theory * 0.95 < empirical < theory * 1.25


def test_plan_costs_attenuation_in_requests():
    ch = JudgeChannel(accuracy=0.85, tie_rate=0.10)
    p = plan(0.5, 0.55, alpha=0.05, beta=0.10, channel=ch, sampling_rate=0.20)
    assert p["observed_p1"] == pytest.approx(0.535)
    assert p["requests_per_verdict"] == pytest.approx(5.556, abs=0.01)
    # The same test through a sharper judge should need fewer verdicts.
    sharp = plan(0.5, 0.55, 0.05, 0.10, JudgeChannel(accuracy=0.95, tie_rate=0.10), 0.20)
    assert sharp["expected_verdicts_under_h1"] < p["expected_verdicts_under_h1"]
