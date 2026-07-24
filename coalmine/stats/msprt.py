"""Mixture SPRT: the always-valid promotion gate.

Wald's SPRT needs a point alternative (p1) chosen in advance; the mixture SPRT
(the method Optimizely industrialized) integrates the likelihood ratio over a
prior on the alternative instead. The resulting mixture martingale, by Ville's
inequality, crosses 1/alpha with probability <= alpha under H0 no matter how
long you watch — so the test is valid at every sample size with no point
alternative to tune.

This implementation is one-sided for the promotion decision: H0: p = p0 vs
H1: p > p0, with a Beta(prior_a, prior_b) prior truncated to (p0, 1). The
truncated-Beta mixture stays in closed form via the regularized incomplete
beta function:

    log L_n = [ln B(a+S, b+F) + ln I-bar_{p0}(a+S, b+F)]
            - [ln B(a, b)     + ln I-bar_{p0}(a, b)]
            - [S ln p0 + F ln(1 - p0)]

where S wins / F losses and I-bar_x(u, v) is the upper tail mass of Beta(u, v)
above x. Reject (promote) when log L_n >= ln(1/alpha). The test never accepts
H0 — absence of a rejection is "no evidence yet", which is exactly the anytime
semantics production monitoring needs.

Like every test in coalmine, p0 lives on the OBSERVED (post-judge-channel)
scale.
"""

import math
from dataclasses import dataclass, field

from scipy.special import betainc, betaln


def _log_upper_mass(a: float, b: float, p0: float) -> float:
    """log of the Beta(a, b) mass above p0, computed from the complement side
    for precision when the mass is tiny."""
    mass = float(betainc(b, a, 1.0 - p0))
    return math.log(mass) if mass > 0.0 else float("-inf")


@dataclass
class MixtureSPRT:
    p0: float
    alpha: float = 0.05
    prior_a: float = 1.0
    prior_b: float = 1.0

    successes: int = field(init=False, default=0)
    n: int = field(init=False, default=0)
    log_lambda: float = field(init=False, default=0.0)
    rejected: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.p0 < 1.0:
            raise ValueError("p0 must be in (0, 1)")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.prior_a <= 0.0 or self.prior_b <= 0.0:
            raise ValueError("prior parameters must be positive")
        self.log_threshold = math.log(1.0 / self.alpha)
        self._log_prior_norm = betaln(self.prior_a, self.prior_b) + _log_upper_mass(
            self.prior_a, self.prior_b, self.p0
        )

    def update(self, win: bool) -> bool:
        """Consume one decisive verdict. Returns True once H0 is rejected
        (challenger promoted); the decision is sticky."""
        if self.rejected:
            return True
        self.successes += bool(win)
        self.n += 1
        s, f = self.successes, self.n - self.successes
        self.log_lambda = (
            betaln(self.prior_a + s, self.prior_b + f)
            + _log_upper_mass(self.prior_a + s, self.prior_b + f, self.p0)
            - self._log_prior_norm
            - s * math.log(self.p0)
            - f * math.log(1.0 - self.p0)
        )
        if self.log_lambda >= self.log_threshold:
            self.rejected = True
        return self.rejected

    def reset(self) -> None:
        self.successes = 0
        self.n = 0
        self.log_lambda = 0.0
        self.rejected = False
