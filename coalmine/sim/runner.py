"""Vectorized Monte Carlo harness.

Runs thousands of SPRT/CUSUM trials as numpy array operations instead of
Python loops over observations. tests/test_runner.py cross-validates these
against the scalar implementations in coalmine.stats on identical streams —
the vectorized code must agree exactly, trial for trial.
"""

import math

import numpy as np
from scipy.special import betainc, betaln
from scipy.stats import norm

from coalmine.stats.confseq import stitched_radius
from coalmine.stats.msprt import _log_upper_mass
from coalmine.stats.wald import increments, thresholds


def _first_crossing(hits: np.ndarray, sentinel: int) -> np.ndarray:
    """Index of first True per row, or sentinel if the row never hits."""
    return np.where(hits.any(axis=1), hits.argmax(axis=1), sentinel)


def run_sprt_batch(
    wins: np.ndarray,
    p0: float,
    p1: float,
    alpha: float = 0.05,
    beta: float = 0.10,
    chunk: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an SPRT down each row of a (trials, n) win matrix.

    Returns (decisions, stop_times): decisions is +1 for accept-H1, -1 for
    accept-H0, 0 for censored (no decision within n); stop_times counts
    observations consumed (= n when censored).
    """
    upper, lower = thresholds(alpha, beta)
    l_w, l_l = increments(p0, p1)
    n_trials, n = wins.shape
    decisions = np.zeros(n_trials, dtype=np.int8)
    stops = np.full(n_trials, n, dtype=np.int64)
    for s in range(0, n_trials, chunk):
        w = wins[s : s + chunk]
        llr = np.cumsum(np.where(w, l_w, l_l), axis=1)
        t_up = _first_crossing(llr >= upper, n)
        t_dn = _first_crossing(llr <= lower, n)
        t_stop = np.minimum(t_up, t_dn)
        decided = t_stop < n
        decisions[s : s + chunk] = np.where(decided, np.where(t_up < t_dn, 1, -1), 0)
        stops[s : s + chunk] = np.where(decided, t_stop + 1, n)
    return decisions, stops


def run_cusum_batch(wins: np.ndarray, p0: float, p1: float, h: float) -> np.ndarray:
    """Run a CUSUM down each row. Returns alarm times (observations consumed
    at first alarm), or -1 for rows that never alarm."""
    l_w, l_l = increments(p0, p1)
    n_trials, n = wins.shape
    stat = np.zeros(n_trials)
    alarm = np.full(n_trials, -1, dtype=np.int64)
    for t in range(n):
        stat = np.maximum(0.0, stat + np.where(wins[:, t], l_w, l_l))
        hit = (alarm < 0) & (stat >= h)
        alarm[hit] = t + 1
    return alarm


def cusum_max_stat(wins: np.ndarray, p0: float, p1: float) -> np.ndarray:
    """Running maximum of the CUSUM statistic per trial, for threshold calibration."""
    l_w, l_l = increments(p0, p1)
    n_trials, n = wins.shape
    stat = np.zeros(n_trials)
    max_stat = np.zeros(n_trials)
    for t in range(n):
        stat = np.maximum(0.0, stat + np.where(wins[:, t], l_w, l_l))
        max_stat = np.maximum(max_stat, stat)
    return max_stat


def calibrate_cusum_threshold(
    null_wins: np.ndarray, p0: float, p1: float, false_alarm_rate: float
) -> float:
    """Threshold h such that P(alarm within the horizon | no change) ~= false_alarm_rate.

    The alarm fires iff the running max crosses h, so h is the (1 - rate)
    quantile of the null running-max distribution.
    """
    max_stats = cusum_max_stat(null_wins, p0, p1)
    return float(np.quantile(max_stats, 1.0 - false_alarm_rate))


def run_msprt_batch(
    wins: np.ndarray,
    p0: float,
    alpha: float = 0.05,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    chunk: int = 250,
) -> np.ndarray:
    """One-sided mixture SPRT down each row of a (trials, n) win matrix.
    Returns rejection times (observations consumed), or -1 if never rejected.
    Must agree trial-for-trial with the scalar MixtureSPRT (cross-validated
    in tests)."""
    n_trials, n = wins.shape
    threshold = math.log(1.0 / alpha)
    steps = np.arange(1, n + 1, dtype=np.float64)
    log_prior_norm = betaln(prior_a, prior_b) + _log_upper_mass(prior_a, prior_b, p0)
    out = np.full(n_trials, -1, dtype=np.int64)
    for s0 in range(0, n_trials, chunk):
        w = wins[s0 : s0 + chunk]
        s = np.cumsum(w, axis=1, dtype=np.float64)
        f = steps[None, :] - s
        with np.errstate(divide="ignore"):
            log_upper = np.log(betainc(prior_b + f, prior_a + s, 1.0 - p0))
        log_lambda = (
            betaln(prior_a + s, prior_b + f)
            + log_upper
            - log_prior_norm
            - s * math.log(p0)
            - f * math.log(1.0 - p0)
        )
        t = _first_crossing(log_lambda >= threshold, n)
        out[s0 : s0 + chunk] = np.where(t == n, -1, t + 1)
    return out


def run_stitched_batch(
    wins: np.ndarray, p0: float, alpha: float = 0.05, chunk: int = 500
) -> np.ndarray:
    """One-sided stitched confidence sequence down each row. Returns rejection
    times, or -1 if p0 is never excluded."""
    n_trials, n = wins.shape
    steps = np.arange(1, n + 1, dtype=np.float64)
    radius = stitched_radius(steps, alpha)
    out = np.full(n_trials, -1, dtype=np.int64)
    for s0 in range(0, n_trials, chunk):
        w = wins[s0 : s0 + chunk]
        p_hat = np.cumsum(w, axis=1, dtype=np.float64) / steps[None, :]
        t = _first_crossing((p_hat - p0) > radius[None, :], n)
        out[s0 : s0 + chunk] = np.where(t == n, -1, t + 1)
    return out


def naive_peeking_false_alarm_rate(
    null_wins: np.ndarray,
    check_every: int = 100,
    min_n: int = 100,
    alpha: float = 0.05,
    chunk: int = 500,
) -> float:
    """The motivating baseline: fraction of null trials where a repeated
    two-sided z-test for win rate != 0.5 rejects at least once.

    Each individual check holds its alpha; checking repeatedly does not. This
    is the false-alarm inflation the sequential methods exist to fix.
    """
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    n_trials, n = null_wins.shape
    checkpoints = np.arange(min_n, n + 1, check_every)
    tripped = 0
    for s in range(0, n_trials, chunk):
        cum = np.cumsum(null_wins[s : s + chunk], axis=1, dtype=np.float64)
        successes = cum[:, checkpoints - 1]
        z = (successes - checkpoints / 2.0) / np.sqrt(checkpoints / 4.0)
        tripped += int((np.abs(z) > z_crit).any(axis=1).sum())
    return tripped / n_trials
