"""The canary lifecycle: shadow → 1% → 5% → 25% → 100%, or rollback.

The capstone state machine. A challenger starts serving 0% of user traffic
(shadow); each ramp step is gated by a fresh non-inferiority test — the
one-sided mixture SPRT rejecting "worse than the margin" (H0: p = 0.5 − δ) at
α/k per gate, so the chance of EVER advancing a truly-worse-than-margin
challenger is ≤ α across the whole ramp, at any peeking cadence. A continuous
CUSUM (threshold calibrated to its own budget over the lifecycle horizon)
watches for regressions the whole way; its alarm rolls the challenger back to
0% from any stage, terminally. Every transition is an event in the log — the
audit trail is the same replayable stream as everything else.

Both tests run on decisive verdicts on the observed (judge-channel) scale,
like every detector in coalmine. The lifecycle is terminal at promoted or
rolled-back: once a challenger is FULL it is the new champion, and continuous
post-promotion monitoring is the standing decision service's job (Phase 5),
not the ramp's.
"""

from dataclasses import dataclass

from coalmine.stats.cusum import CUSUM
from coalmine.stats.msprt import MixtureSPRT

STAGES: tuple[tuple[str, float], ...] = (
    ("shadow", 0.0),
    ("canary-1", 0.01),
    ("canary-5", 0.05),
    ("canary-25", 0.25),
)
FULL_SHARE = 1.0

ADVANCED = "canary_advanced"
PROMOTED = "canary_promoted"
ROLLED_BACK = "canary_rolled_back"


@dataclass(frozen=True)
class Transition:
    kind: str  # ADVANCED | PROMOTED | ROLLED_BACK
    stage: str
    share: float
    request_index: int
    verdicts_in_stage: int


class CanaryLifecycle:
    def __init__(
        self,
        cusum_h: float,
        margin: float = 0.06,
        alpha: float = 0.05,
        cusum_p1: float = 0.45,
    ):
        if not 0.0 < margin < 0.5:
            raise ValueError("margin must be in (0, 0.5)")
        self.margin = margin
        self.gate_alpha = alpha / len(STAGES)  # union bound over the ramp's gates
        self._stage_idx = 0
        self.state = "running"  # running | promoted | rolled_back
        self.gate = self._new_gate()
        self.alarm = CUSUM(p0=0.5, p1=cusum_p1, h=cusum_h)
        self.transitions: list[Transition] = []
        self.verdicts_seen = 0
        self._verdicts_in_stage = 0

    def _new_gate(self) -> MixtureSPRT:
        return MixtureSPRT(p0=0.5 - self.margin, alpha=self.gate_alpha)

    @property
    def stage_name(self) -> str:
        if self.state == "promoted":
            return "full"
        if self.state == "rolled_back":
            return "rolled-back"
        return STAGES[self._stage_idx][0]

    @property
    def share(self) -> float:
        if self.state == "promoted":
            return FULL_SHARE
        if self.state == "rolled_back":
            return 0.0
        return STAGES[self._stage_idx][1]

    @property
    def terminal(self) -> bool:
        return self.state != "running"

    def update(self, win: bool, request_index: int) -> Transition | None:
        """Consume one decisive verdict; returns a Transition when the
        lifecycle changes state (advance, promote, or rollback)."""
        if self.terminal:
            return None
        self.verdicts_seen += 1
        self._verdicts_in_stage += 1
        if self.alarm.update(win):
            self.state = "rolled_back"
            transition = Transition(
                ROLLED_BACK, "rolled-back", 0.0, request_index, self._verdicts_in_stage
            )
            self.transitions.append(transition)
            return transition
        if self.gate.update(win):
            self._stage_idx += 1
            self._verdicts_in_stage = 0
            if self._stage_idx == len(STAGES):
                self.state = "promoted"
                transition = Transition(PROMOTED, "full", FULL_SHARE, request_index, 0)
            else:
                name, share = STAGES[self._stage_idx]
                transition = Transition(ADVANCED, name, share, request_index, 0)
                self.gate = self._new_gate()
            self.transitions.append(transition)
            return transition
        return None
