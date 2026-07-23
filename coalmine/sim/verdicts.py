"""Synthetic decisive-verdict streams for Monte Carlo validation.

The decision engine consumes win/loss verdicts and does not care where they
come from. For the statistics experiments we generate them here: Bernoulli
draws at a known observed win rate, with an optional changepoint injecting a
known regression. Ground truth is exact by construction, which is what makes
detection latency and false-alarm rate measurable claims rather than vibes.

All streams are on the observed (post-judge-channel) scale; experiments map
true effect sizes through JudgeChannel before generating.
"""

import numpy as np


def gen_wins(rng: np.random.Generator, trials: int, n: int, win_rate: float) -> np.ndarray:
    """(trials, n) boolean matrix of iid decisive verdicts at a fixed win rate."""
    return rng.random((trials, n)) < win_rate


def gen_wins_changepoint(
    rng: np.random.Generator,
    trials: int,
    n: int,
    win_rate_before: float,
    win_rate_after: float,
    changepoint: int,
) -> np.ndarray:
    """(trials, n) verdicts where the win rate shifts at index `changepoint`."""
    if not 0 <= changepoint <= n:
        raise ValueError("changepoint must be within the stream")
    p = np.full(n, win_rate_before)
    p[changepoint:] = win_rate_after
    return rng.random((trials, n)) < p[None, :]
