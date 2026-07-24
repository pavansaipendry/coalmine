"""Judge ensemble: majority vote for robustness, disagreement as a drift signal.

Three cheap judges beat one in two distinct ways. Majority voting absorbs a
single member's degradation (the vote stays mostly right while one judge goes
bad). And the *disagreement rate* between members is an early sensor-drift
warning that needs no ground truth at all: healthy members disagree at a
stable baseline rate, so a rising rate flags trouble before anchor-set
calibration (which needs labeled data and scheduled rounds) would run.
"""

from dataclasses import dataclass

import numpy as np

from coalmine.judging.judges import FIRST, SECOND, TIE, PairwiseJudge


@dataclass(frozen=True)
class EnsembleVerdict:
    verdict: str  # "first" | "second" | "tie" — the majority outcome
    member_verdicts: tuple[str, ...]
    disagreed: bool  # any member differed from any other


class EnsembleJudge:
    """PairwiseJudge over k members: majority vote in position space; ties in
    the vote (or explicit majority-tie) return TIE. Each member draws from an
    independent spawn of the per-comparison rng, so members are deterministic
    but uncorrelated."""

    def __init__(self, members: list[PairwiseJudge], judge_id: str = "ensemble"):
        if len(members) < 2:
            raise ValueError("an ensemble needs at least 2 members")
        self.members = members
        self.judge_id = judge_id
        self.disagreements: list[bool] = []

    async def compare(
        self,
        query: str,
        first: dict,
        second: dict,
        rng: np.random.Generator | None = None,
    ) -> str:
        if rng is None:
            raise ValueError("EnsembleJudge needs a per-comparison rng to spawn member streams")
        result = await self.compare_detailed(query, first, second, rng)
        return result.verdict

    async def compare_detailed(
        self,
        query: str,
        first: dict,
        second: dict,
        rng: np.random.Generator,
    ) -> EnsembleVerdict:
        member_rngs = rng.spawn(len(self.members))
        verdicts = tuple(
            [
                await judge.compare(query, first, second, rng=member_rng)
                for judge, member_rng in zip(self.members, member_rngs)
            ]
        )
        counts = {v: verdicts.count(v) for v in (FIRST, SECOND, TIE)}
        majority_bar = len(self.members) // 2 + 1
        if counts[FIRST] >= majority_bar:
            majority = FIRST
        elif counts[SECOND] >= majority_bar:
            majority = SECOND
        else:
            majority = TIE
        disagreed = len(set(verdicts)) > 1
        self.disagreements.append(disagreed)
        return EnsembleVerdict(majority, verdicts, disagreed)


def majority_accuracy(accuracy: float, k: int) -> float:
    """P(majority of k independent judges is correct), each correct with
    probability `accuracy`, no ties. The no-bias reference for ensemble
    detection: at a = 0.85, k = 3 gives 0.939 — higher effective accuracy
    means less judge-channel attenuation, which is why ensembles detect
    regressions faster. (With ties and position bias, the experiments measure
    the effective accuracy empirically instead of trusting this formula.)"""
    from math import comb

    if k % 2 == 0 or k < 1:
        raise ValueError("k must be odd and positive")
    needed = k // 2 + 1
    return float(
        sum(comb(k, i) * accuracy**i * (1 - accuracy) ** (k - i) for i in range(needed, k + 1))
    )


def rolling_disagreement(disagreements: list[bool], window: int) -> np.ndarray:
    """Rolling mean of the disagreement indicator — the sensor-drift signal."""
    if len(disagreements) < window:
        raise ValueError(f"need at least {window} comparisons")
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(disagreements, dtype=float), kernel, mode="valid")


def calibrate_disagreement_threshold(
    baseline_rate: float,
    window: int,
    n_checks: int,
    false_alarm_rate: float,
    trials: int = 2_000,
    seed: int = 0,
) -> float:
    """(1 - rate) quantile of the max rolling disagreement under a healthy
    ensemble (iid Bernoulli at the measured baseline rate) — the same
    max-statistic calibration used everywhere else in coalmine."""
    rng = np.random.default_rng(seed)
    horizon = window + n_checks - 1
    kernel = np.ones(window) / window
    max_stat = np.zeros(trials)
    for t in range(trials):
        stream = rng.random(horizon) < baseline_rate
        rolled = np.convolve(stream.astype(float), kernel, mode="valid")
        max_stat[t] = rolled.max()
    return float(np.quantile(max_stat, 1.0 - false_alarm_rate))
