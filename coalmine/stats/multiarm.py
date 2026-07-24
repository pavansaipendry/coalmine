"""k challengers, one champion: family-wise error control.

Running k promotion tests each at alpha means the chance of promoting SOME
not-actually-better challenger grows with k — the multi-arm version of the
peeking problem. The fix composes cleanly with anytime validity: give each
arm's always-valid test an alpha/k budget and the union bound caps the
family-wise error at alpha over the entire monitoring horizon, at any k, with
no coordination between arms. The Phase 4 experiment measures both the
protection and its price (slower per-arm decisions).
"""

from coalmine.stats.msprt import MixtureSPRT


class MultiArmMonitor:
    def __init__(
        self,
        arm_ids: list[str],
        p0: float = 0.5,
        alpha: float = 0.05,
        corrected: bool = True,
    ):
        if not arm_ids or len(set(arm_ids)) != len(arm_ids):
            raise ValueError("arm_ids must be non-empty and unique")
        self.alpha = alpha
        self.per_arm_alpha = alpha / len(arm_ids) if corrected else alpha
        self.tests = {arm: MixtureSPRT(p0, self.per_arm_alpha) for arm in arm_ids}
        self.promoted: list[str] = []

    def update(self, arm_id: str, win: bool) -> bool:
        """Consume one decisive verdict for one arm. Returns True iff this
        verdict newly promoted the arm."""
        test = self.tests[arm_id]
        already = test.rejected
        rejected = test.update(win)
        newly = rejected and not already
        if newly:
            self.promoted.append(arm_id)
        return newly
