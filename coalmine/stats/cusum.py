"""Page's CUSUM: detect a drop in win rate that begins at an unknown time.

A vanilla SPRT assumes the win rate is fixed for the whole stream; run from
t=0 it will happily accept H0 long before a late regression begins. Page's
test is the SPRT reflected at zero: the statistic accumulates evidence for
H1 (win rate p1 < p0) but never goes negative, so it stays primed to detect a
change starting at any point. This is the regression-alarm primitive; the
plain SPRT is the promotion gate.

The alarm threshold h is calibrated by simulation to a target false-alarm
probability over a monitoring horizon (see coalmine.sim.runner).
"""

import math
from dataclasses import dataclass, field


@dataclass
class CUSUM:
    p0: float  # in-control win rate (observed scale)
    p1: float  # out-of-control win rate the test is tuned to detect, p1 < p0
    h: float  # alarm threshold, > 0

    stat: float = field(init=False, default=0.0)
    n: int = field(init=False, default=0)
    alarmed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.p1 < self.p0 < 1.0):
            raise ValueError("need 0 < p1 < p0 < 1: CUSUM here detects win-rate drops")
        if self.h <= 0.0:
            raise ValueError("h must be positive")
        self.llr_win = math.log(self.p1 / self.p0)
        self.llr_loss = math.log((1.0 - self.p1) / (1.0 - self.p0))

    def update(self, win: bool) -> bool:
        """Consume one decisive verdict. Returns True once the alarm has fired."""
        if self.alarmed:
            return True
        self.stat = max(0.0, self.stat + (self.llr_win if win else self.llr_loss))
        self.n += 1
        if self.stat >= self.h:
            self.alarmed = True
        return self.alarmed

    def reset(self) -> None:
        self.stat = 0.0
        self.n = 0
        self.alarmed = False
