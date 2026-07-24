"""Time-uniform confidence sequence (Howard et al. stitched boundary).

The modern always-valid contrast to the mixture SPRT: a closed-form boundary,
no prior to choose, valid simultaneously at every n. For a [0, 1]-bounded
(sigma = 1/2 sub-Gaussian) mean, the stitched radius

    r_n = 1.7 * sigma * sqrt((ln ln(2n) + 0.72 ln(5.2 / alpha)) / n)

satisfies P(exists n: |p-hat_n - p| > r_n) <= alpha. Used one-sided here for
the promotion decision: reject H0: p = p0 when p-hat_n - p0 > r_n(alpha).

The stitched constants trade tightness for closed form — the boundary is
conservative, and the Phase 4 comparison quantifies exactly how much decision
latency that costs versus the mixture SPRT. Realized error rates of every
method are measured by Monte Carlo rather than trusted from formulas.
"""

from dataclasses import dataclass, field

import numpy as np


def stitched_radius(n, alpha: float, sigma: float = 0.5):
    """Time-uniform radius; accepts scalar or array n (n >= 1)."""
    n = np.asarray(n, dtype=np.float64)
    return 1.7 * sigma * np.sqrt((np.log(np.log(2.0 * n)) + 0.72 * np.log(5.2 / alpha)) / n)


@dataclass
class StitchedCS:
    p0: float
    alpha: float = 0.05

    successes: int = field(init=False, default=0)
    n: int = field(init=False, default=0)
    rejected: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.p0 < 1.0:
            raise ValueError("p0 must be in (0, 1)")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

    def update(self, win: bool) -> bool:
        """Consume one decisive verdict; sticky rejection when the lower
        confidence bound clears p0."""
        if self.rejected:
            return True
        self.successes += bool(win)
        self.n += 1
        p_hat = self.successes / self.n
        if p_hat - self.p0 > float(stitched_radius(self.n, self.alpha)):
            self.rejected = True
        return self.rejected

    def reset(self) -> None:
        self.successes = 0
        self.n = 0
        self.rejected = False


def radius_table(alpha: float, ns: list[int]) -> dict[int, float]:
    """Convenience for docs/tests: the boundary at chosen sample sizes."""
    return {n: float(stitched_radius(n, alpha)) for n in ns}
