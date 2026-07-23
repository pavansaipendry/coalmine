"""Wald's closed-form planning approximations for the Bernoulli SPRT.

Design-time formulas: given a test (p0, p1, alpha, beta) and the win rate p
actually in effect, predict the probability of accepting H1 and the expected
number of observations to a decision. These power the planning calculator
(how long will a decision take at this sampling rate and judge accuracy?) and
serve as the theory curve the Monte Carlo experiments must reproduce.

Approximations ignore boundary overshoot, so empirical error rates run
slightly conservative and empirical E[N] slightly above these curves.
"""

import math

from scipy.optimize import brentq

from coalmine.stats.channel import JudgeChannel


def thresholds(alpha: float, beta: float) -> tuple[float, float]:
    return math.log((1.0 - beta) / alpha), math.log(beta / (1.0 - alpha))


def increments(p0: float, p1: float) -> tuple[float, float]:
    return math.log(p1 / p0), math.log((1.0 - p1) / (1.0 - p0))


def accept_h1_probability(p: float, p0: float, p1: float, alpha: float, beta: float) -> float:
    """P(test accepts H1) when observations are Bernoulli(p)."""
    upper, lower = thresholds(alpha, beta)
    l_w, l_l = increments(p0, p1)
    drift = p * l_w + (1.0 - p) * l_l
    r1 = p1 / p0
    r0 = (1.0 - p1) / (1.0 - p0)

    # h != 0 solving E_p[(r)^h] = 1; h -> 0 exactly when drift = 0.
    def f(h: float) -> float:
        return p * r1**h + (1.0 - p) * r0**h - 1.0

    zero_drift_answer = -lower / (upper - lower)
    if abs(drift) < 1e-12:
        return zero_drift_answer
    lo, hi = (1e-4, 200.0) if drift < 0 else (-200.0, -1e-4)
    try:
        h = brentq(f, lo, hi)
    except ValueError:
        # Root lies inside (0, 1e-4): drift is negligible, use the limit.
        return zero_drift_answer
    a_h, b_h = math.exp(upper * h), math.exp(lower * h)
    return (1.0 - b_h) / (a_h - b_h)


def expected_n(p: float, p0: float, p1: float, alpha: float, beta: float) -> float:
    """Expected observations to a decision when observations are Bernoulli(p)."""
    upper, lower = thresholds(alpha, beta)
    l_w, l_l = increments(p0, p1)
    drift = p * l_w + (1.0 - p) * l_l
    if abs(drift) < 1e-9:
        second_moment = p * l_w**2 + (1.0 - p) * l_l**2
        return -(lower * upper) / second_moment
    p_h1 = accept_h1_probability(p, p0, p1, alpha, beta)
    return (p_h1 * upper + (1.0 - p_h1) * lower) / drift


def plan(
    true_p0: float,
    true_p1: float,
    alpha: float,
    beta: float,
    channel: JudgeChannel,
    sampling_rate: float,
) -> dict:
    """Planning calculator: design a test on the observed scale and cost it in live requests.

    Answers "at this judge sampling rate, judge accuracy, and minimum detectable
    effect, how much traffic does a decision take?" for both hypotheses.
    """
    obs_p0 = channel.observed_win_rate(true_p0)
    obs_p1 = channel.observed_win_rate(true_p1)
    rpv = channel.requests_per_verdict(sampling_rate)
    n_h0 = expected_n(obs_p0, obs_p0, obs_p1, alpha, beta)
    n_h1 = expected_n(obs_p1, obs_p0, obs_p1, alpha, beta)
    return {
        "observed_p0": obs_p0,
        "observed_p1": obs_p1,
        "attenuation": channel.attenuation(),
        "requests_per_verdict": rpv,
        "expected_verdicts_under_h0": n_h0,
        "expected_verdicts_under_h1": n_h1,
        "expected_requests_under_h0": n_h0 * rpv,
        "expected_requests_under_h1": n_h1 * rpv,
    }
