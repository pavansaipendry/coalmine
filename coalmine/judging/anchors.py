"""Anchor sets: the judge's own calibration instrument.

A frozen set of (query, response pair, known label) items, re-judged
periodically. Judge agreement with the anchors estimates judge accuracy — the
channel parameter every sequential test is designed against. If the estimate's
confidence interval drops below the design tolerance, the system's *sensor*
has degraded and detection guarantees no longer hold: that is itself an alarm,
independent of any traffic-level signal.

Synthetic anchors are built from the good/bad response pools (label known by
construction); the same format holds MT-Bench human-judgment imports later.
"""

import math
from dataclasses import dataclass

import numpy as np

from coalmine.judging.judges import PairwiseJudge, compare_randomized
from coalmine.traffic.dataset import Query
from coalmine.traffic.pools import ResponsePool


@dataclass(frozen=True)
class AnchorItem:
    id: str
    query_text: str
    text_a: str
    text_b: str
    label: str  # "a" | "b" — which response is truly better


@dataclass(frozen=True)
class CalibrationResult:
    n_items: int
    n_decisive: int
    n_correct: int
    accuracy: float
    wilson_low: float
    wilson_high: float


def build_synthetic_anchors(
    queries: list[Query], pool: ResponsePool, n: int, seed: int
) -> list[AnchorItem]:
    """n anchor items: one good and one bad response per sampled query, with
    the good side assigned by a seeded coin so labels are balanced."""
    items = []
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        q = queries[int(rng.integers(len(queries)))]
        good = pool.responses(q.id, "good")
        bad = pool.responses(q.id, "bad")
        good_text = good[int(rng.integers(len(good)))]
        bad_text = bad[int(rng.integers(len(bad)))]
        if rng.random() < 0.5:
            items.append(AnchorItem(f"anchor-{i}", q.text, good_text, bad_text, "a"))
        else:
            items.append(AnchorItem(f"anchor-{i}", q.text, bad_text, good_text, "b"))
    return items


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        raise ValueError("empty sample")
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


async def calibrate_judge(
    judge: PairwiseJudge,
    anchors: list[AnchorItem],
    seed: int,
    round_index: int = 0,
) -> CalibrationResult:
    """Re-judge the anchor set (randomized presentation) and estimate judge
    accuracy with a Wilson interval. Ties are excluded from the denominator."""
    n_correct = 0
    n_decisive = 0
    for i, item in enumerate(anchors):
        rng = np.random.default_rng([seed, round_index, i])
        a_pool = "good" if item.label == "a" else "bad"
        b_pool = "bad" if item.label == "a" else "good"
        result = await compare_randomized(
            judge,
            item.query_text,
            {"text": item.text_a, "source_pool": a_pool},
            {"text": item.text_b, "source_pool": b_pool},
            rng,
        )
        if result.winner == "tie":
            continue
        n_decisive += 1
        n_correct += result.winner == item.label
    if n_decisive == 0:
        raise ValueError("judge tied on every anchor — no accuracy signal")
    low, high = wilson_interval(n_correct, n_decisive)
    return CalibrationResult(
        n_items=len(anchors),
        n_decisive=n_decisive,
        n_correct=n_correct,
        accuracy=n_correct / n_decisive,
        wilson_low=low,
        wilson_high=high,
    )
