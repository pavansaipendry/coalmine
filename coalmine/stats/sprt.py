"""Wald's Sequential Probability Ratio Test for Bernoulli verdict streams.

The promotion gate: H0 (challenger no better, win rate p0) vs H1 (challenger
better, win rate p1 > p0), observed one pairwise verdict at a time. Unlike a
repeated t-test, the SPRT may be checked after every observation while keeping
its error rates: P(accept H1 | H0) <= alpha, P(accept H0 | H1) <= beta, up to
Wald's boundary-overshoot approximation.

Both p0 and p1 must be on the OBSERVED scale — pass true win rates through
JudgeChannel.observed_win_rate first.
"""

import math
from dataclasses import dataclass, field
from enum import Enum


class Decision(Enum):
    CONTINUE = "continue"
    ACCEPT_H1 = "accept_h1"  # challenger wins at rate p1 — promote
    ACCEPT_H0 = "accept_h0"  # challenger wins at rate p0 — reject


@dataclass
class SPRT:
    p0: float
    p1: float
    alpha: float = 0.05  # P(accept H1 | H0)
    beta: float = 0.10  # P(accept H0 | H1)

    llr: float = field(init=False, default=0.0)
    n: int = field(init=False, default=0)
    decision: Decision = field(init=False, default=Decision.CONTINUE)

    def __post_init__(self) -> None:
        if not (0.0 < self.p0 < 1.0 and 0.0 < self.p1 < 1.0 and self.p0 != self.p1):
            raise ValueError("p0 and p1 must be distinct probabilities in (0, 1)")
        if not (0.0 < self.alpha < 1.0 and 0.0 < self.beta < 1.0):
            raise ValueError("alpha and beta must be in (0, 1)")
        self.upper = math.log((1.0 - self.beta) / self.alpha)
        self.lower = math.log(self.beta / (1.0 - self.alpha))
        self.llr_win = math.log(self.p1 / self.p0)
        self.llr_loss = math.log((1.0 - self.p1) / (1.0 - self.p0))

    def update(self, win: bool) -> Decision:
        """Consume one decisive verdict (True = challenger won). Ties are dropped upstream."""
        if self.decision is not Decision.CONTINUE:
            return self.decision
        self.llr += self.llr_win if win else self.llr_loss
        self.n += 1
        if self.llr >= self.upper:
            self.decision = Decision.ACCEPT_H1
        elif self.llr <= self.lower:
            self.decision = Decision.ACCEPT_H0
        return self.decision

    def reset(self) -> None:
        """Restart the test, e.g. after an input-drift alarm invalidates the stream."""
        self.llr = 0.0
        self.n = 0
        self.decision = Decision.CONTINUE
