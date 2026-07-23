"""Pairwise judges and position randomization.

Judges see two responses in *position* space ("first"/"second") and never know
which config produced which. The comparator flips presentation order with a
seeded coin and maps the verdict back to config space — so position bias adds
noise but no mean shift, and because the presented order is recorded on every
verdict, the bias itself is measurable from the event log.

NoisyOracleJudge is the simulation sensor: it knows ground truth (source_pool)
and corrupts it with exactly the noise model the decision engine is designed
against — finite accuracy, ties, and a preference for the first-shown
response. Real LLM judges implement the same protocol (coalmine.judging.llm)
and ignore the simulation-only fields.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

FIRST, SECOND, TIE = "first", "second", "tie"


class PairwiseJudge(Protocol):
    judge_id: str

    async def compare(
        self,
        query: str,
        first: dict,
        second: dict,
        rng: np.random.Generator | None = None,
    ) -> str:
        """Return "first", "second", or "tie". `rng` drives simulation judges
        and is ignored by real ones."""
        ...


@dataclass(frozen=True)
class ComparisonResult:
    winner: str  # "a" | "b" | "tie" — config space
    first_shown: str  # "a" | "b" — which config was presented first


async def compare_randomized(
    judge: PairwiseJudge,
    query: str,
    a: dict,
    b: dict,
    rng: np.random.Generator,
    randomize: bool = True,
) -> ComparisonResult:
    """Run one pairwise comparison with (optionally) randomized presentation
    order and map the position-space verdict back to config space.

    randomize=False always shows `a` first — kept only so experiments can
    demonstrate what position bias does to an unrandomized pipeline.
    """
    swap = randomize and bool(rng.random() < 0.5)
    first, second = (b, a) if swap else (a, b)
    position_verdict = await judge.compare(query, first, second, rng=rng)
    if position_verdict == TIE:
        winner = "tie"
    elif position_verdict == FIRST:
        winner = "b" if swap else "a"
    else:
        winner = "a" if swap else "b"
    return ComparisonResult(winner=winner, first_shown="b" if swap else "a")


@dataclass(frozen=True)
class NoisyOracleJudge:
    """Simulation judge: ground truth corrupted by a known noise model.

    Draw order per comparison: tie (tie_rate) -> blind first-pick
    (position_bias) -> accuracy-weighted pick of the truly better response
    (equal-quality pairs are a fair coin). Effective accuracy on unequal pairs
    is position_bias * 0.5 + (1 - position_bias) * accuracy.
    """

    accuracy: float = 0.85
    tie_rate: float = 0.10
    position_bias: float = 0.0
    judge_id: str = "noisy-oracle"

    def __post_init__(self) -> None:
        if not 0.5 < self.accuracy <= 1.0:
            raise ValueError("accuracy must be in (0.5, 1.0]")
        for name in ("tie_rate", "position_bias"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")

    async def compare(
        self,
        query: str,
        first: dict,
        second: dict,
        rng: np.random.Generator | None = None,
    ) -> str:
        if rng is None:
            raise ValueError("NoisyOracleJudge needs a per-comparison rng")
        if rng.random() < self.tie_rate:
            return TIE
        if self.position_bias and rng.random() < self.position_bias:
            return FIRST
        rank = {"good": 1, "bad": 0}
        q_first, q_second = rank[first["source_pool"]], rank[second["source_pool"]]
        if q_first == q_second:
            return FIRST if rng.random() < 0.5 else SECOND
        truly_better = FIRST if q_first > q_second else SECOND
        other = SECOND if truly_better == FIRST else FIRST
        return truly_better if rng.random() < self.accuracy else other
