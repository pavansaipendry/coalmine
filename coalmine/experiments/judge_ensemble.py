"""Phase 5 experiment: the ensemble absorbs a failing judge — and notices it.

Three oracle judges (accuracy 0.85) vote on 6,000 good-vs-bad comparisons.
At comparison 3,000 one member silently degrades to 0.55 — near coin-flip.
Two things happen, both measured: the majority vote's accuracy barely moves
(the ensemble absorbs the failure), and the members' disagreement rate jumps
immediately — a sensor-drift signal that needs no ground-truth labels at all,
firing long before the next scheduled anchor-set calibration round would.
The disagreement threshold is calibrated to a 5% false-alarm budget over the
monitoring horizon, same discipline as every other detector here.

Run:  python -m coalmine.experiments.judge_ensemble
"""

import argparse
import asyncio
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.experiments import style
from coalmine.judging.ensemble import (
    EnsembleJudge,
    calibrate_disagreement_threshold,
    rolling_disagreement,
)
from coalmine.judging.judges import NoisyOracleJudge, compare_randomized
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool

N_COMPARISONS = 6_000
DEGRADE_AT = 3_000
HEALTHY_ACCURACY = 0.85
DEGRADED_ACCURACY = 0.55
TIE_RATE = 0.10
WINDOW = 300


def _members(third_accuracy: float):
    return [
        NoisyOracleJudge(accuracy=HEALTHY_ACCURACY, tie_rate=TIE_RATE, judge_id="m0"),
        NoisyOracleJudge(accuracy=HEALTHY_ACCURACY, tie_rate=TIE_RATE, judge_id="m1"),
        NoisyOracleJudge(accuracy=third_accuracy, tie_rate=TIE_RATE, judge_id="m2"),
    ]


async def run(seed: int) -> dict:
    queries = synthetic_queries(300)
    pool = ResponsePool.build_synthetic(queries)
    healthy = EnsembleJudge(_members(HEALTHY_ACCURACY))
    degraded = EnsembleJudge(_members(DEGRADED_ACCURACY))

    disagreements: list[bool] = []
    majority_correct = {"pre": [], "post": []}
    member_correct = {"pre": [], "post": []}
    for i in range(N_COMPARISONS):
        rng = np.random.default_rng([seed, i])
        q = queries[i % len(queries)]
        good = pool.responses(q.id, "good")[0]
        bad = pool.responses(q.id, "bad")[0]
        ensemble = healthy if i < DEGRADE_AT else degraded
        phase = "pre" if i < DEGRADE_AT else "post"
        member_rng = np.random.default_rng([seed, i, 999])
        result = await compare_randomized(
            ensemble,
            q.text,
            {"text": good, "source_pool": "good"},
            {"text": bad, "source_pool": "bad"},
            rng,
        )
        disagreements.append(ensemble.disagreements[-1])
        if result.winner != "tie":
            majority_correct[phase].append(result.winner == "a")
        solo = await _members(HEALTHY_ACCURACY if phase == "pre" else DEGRADED_ACCURACY)[
            2
        ].compare(q.text, {"source_pool": "good"}, {"source_pool": "bad"}, rng=member_rng)
        if solo != "tie":
            member_correct[phase].append(solo == "first")

    baseline_rate = float(np.mean(disagreements[:DEGRADE_AT]))
    threshold = calibrate_disagreement_threshold(
        baseline_rate, WINDOW, N_COMPARISONS - WINDOW + 1, 0.05, trials=2_000, seed=seed
    )
    rolled = rolling_disagreement(disagreements, WINDOW)
    crossings = np.nonzero(rolled >= threshold)[0]
    alarm_at = int(crossings[0] + WINDOW) if len(crossings) else None

    results = {
        "config": {
            "n_comparisons": N_COMPARISONS,
            "degrade_at": DEGRADE_AT,
            "accuracies": [HEALTHY_ACCURACY, DEGRADED_ACCURACY],
            "tie_rate": TIE_RATE,
            "window": WINDOW,
            "seed": seed,
        },
        "baseline_disagreement": baseline_rate,
        "post_disagreement": float(np.mean(disagreements[DEGRADE_AT:])),
        "threshold": threshold,
        "alarm_at": alarm_at,
        "alarm_delay": (alarm_at - DEGRADE_AT) if alarm_at else None,
        "majority_accuracy": {k: float(np.mean(v)) for k, v in majority_correct.items()},
        "member_accuracy": {k: float(np.mean(v)) for k, v in member_correct.items()},
        "rolling": rolled.tolist(),
    }
    print(
        f"  disagreement {baseline_rate:.3f} -> {results['post_disagreement']:.3f} · "
        f"alarm @ {alarm_at} (+{results['alarm_delay']} after degradation)"
    )
    print(
        f"  majority accuracy {results['majority_accuracy']['pre']:.3f} -> "
        f"{results['majority_accuracy']['post']:.3f} while the member fell to "
        f"{results['member_accuracy']['post']:.3f}"
    )
    return results


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    cfg = results["config"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    style.style_axes(ax)
    rolled = results["rolling"]
    x = np.arange(cfg["window"], cfg["window"] + len(rolled))
    ax.plot(x, rolled, color=style.SERIES_1, linewidth=1.5, label="Rolling disagreement rate")
    ax.axhline(
        results["threshold"], color=style.BASELINE, linewidth=1.0, linestyle=(0, (4, 3))
    )
    ax.text(
        cfg["window"] + 60, results["threshold"] + 0.005, "calibrated threshold",
        color=style.MUTED, fontsize=8.5,
    )
    ax.axvline(cfg["degrade_at"], color=style.GRID, linewidth=1.0)
    ax.text(
        cfg["degrade_at"] - 60, 0.75, "one member degrades\n(0.85 → 0.55)",
        color=style.INK_2, fontsize=8.5, ha="right",
    )
    alarm = results["alarm_at"]
    if alarm:
        y = rolled[alarm - cfg["window"]]
        ax.plot([alarm], [y], marker="o", markersize=7, color=style.SERIES_1)
        ax.annotate(
            f"sensor alarm @ {alarm:,}\n(+{results['alarm_delay']} comparisons)",
            (alarm, y),
            textcoords="offset points",
            xytext=(12, -6),
            color=style.INK,
            fontsize=9,
            fontweight="bold",
        )
    maj = results["majority_accuracy"]
    ax.set_xlabel("Comparison index")
    ax.set_ylabel("Member disagreement rate (rolling)")
    style.titles(
        ax,
        "Disagreement flags a failing judge — no labels needed",
        f"3-judge ensemble · majority accuracy held {maj['pre']:.1%} → {maj['post']:.1%} "
        f"while the degraded member fell to {results['member_accuracy']['post']:.1%}\n"
        f"threshold calibrated to a 5% false-alarm budget over the horizon "
        f"(window {cfg['window']})",
    )
    ax.legend(frameon=False, loc="upper left", labelcolor=style.INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "judge_ensemble.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = asyncio.run(run(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "judge_ensemble.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
