"""Phase 3 experiment: the system monitors its own sensor.

Every detection guarantee is conditioned on the judge's accuracy — the channel
parameter the tests are designed against. This experiment re-judges a frozen
anchor set each round while the judge's true accuracy silently degrades
(0.85 → 0.72 at round 10, simulating judge-model drift). The Wilson interval
around measured accuracy crosses the design tolerance within a round or two of
the degradation: the monitoring system alarms on its own failing sensor before
the sensor quietly invalidates every downstream decision.

Run:  python -m coalmine.experiments.judge_calibration
"""

import argparse
import asyncio
import json
from pathlib import Path

import matplotlib.pyplot as plt

from coalmine.experiments import style
from coalmine.judging.anchors import build_synthetic_anchors, calibrate_judge
from coalmine.judging.judges import NoisyOracleJudge
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool

N_ANCHORS = 300
N_ROUNDS = 20
DEGRADE_AT = 10
ACCURACY_HEALTHY = 0.85
ACCURACY_DEGRADED = 0.72
TIE_RATE = 0.10
DESIGN_ACCURACY = 0.85
TOLERANCE_FLOOR = 0.80  # alarm when the CI upper bound falls below this


async def run(seed: int) -> dict:
    queries = synthetic_queries(200)
    pool = ResponsePool.build_synthetic(queries)
    anchors = build_synthetic_anchors(queries, pool, N_ANCHORS, seed)

    rounds = []
    alarm_round = None
    for r in range(N_ROUNDS):
        true_accuracy = ACCURACY_HEALTHY if r < DEGRADE_AT else ACCURACY_DEGRADED
        judge = NoisyOracleJudge(accuracy=true_accuracy, tie_rate=TIE_RATE)
        cal = await calibrate_judge(judge, anchors, seed, round_index=r)
        alarmed = cal.wilson_high < TOLERANCE_FLOOR
        if alarmed and alarm_round is None:
            alarm_round = r
        rounds.append(
            {
                "round": r,
                "true_accuracy": true_accuracy,
                "measured": cal.accuracy,
                "wilson_low": cal.wilson_low,
                "wilson_high": cal.wilson_high,
                "n_decisive": cal.n_decisive,
                "alarmed": alarmed,
            }
        )
        flag = "  << SENSOR ALARM" if alarmed else ""
        print(
            f"  round {r:2d}: true {true_accuracy:.2f}, measured {cal.accuracy:.3f} "
            f"[{cal.wilson_low:.3f}, {cal.wilson_high:.3f}]{flag}"
        )

    return {
        "config": {
            "n_anchors": N_ANCHORS,
            "n_rounds": N_ROUNDS,
            "degrade_at_round": DEGRADE_AT,
            "accuracy_schedule": [ACCURACY_HEALTHY, ACCURACY_DEGRADED],
            "tie_rate": TIE_RATE,
            "design_accuracy": DESIGN_ACCURACY,
            "tolerance_floor": TOLERANCE_FLOOR,
            "seed": seed,
        },
        "rounds": rounds,
        "alarm_round": alarm_round,
        "rounds_to_alarm": (alarm_round - DEGRADE_AT) if alarm_round is not None else None,
    }


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    rounds = results["rounds"]
    x = [r["round"] for r in rounds]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    style.style_axes(ax)
    ax.fill_between(
        x,
        [r["wilson_low"] for r in rounds],
        [r["wilson_high"] for r in rounds],
        color=style.BAND_1,
        alpha=0.55,
        linewidth=0,
        label="95% Wilson interval",
    )
    ax.plot(
        x,
        [r["measured"] for r in rounds],
        color=style.SERIES_1,
        linewidth=1.8,
        marker="o",
        markersize=5,
        label="Measured accuracy",
    )
    ax.axhline(DESIGN_ACCURACY, color=style.BASELINE, linewidth=1.0)
    ax.text(0.1, DESIGN_ACCURACY + 0.004, "design accuracy", color=style.MUTED, fontsize=8.5)
    ax.axhline(TOLERANCE_FLOOR, color=style.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(0.1, TOLERANCE_FLOOR + 0.004, "tolerance floor", color=style.MUTED, fontsize=8.5)
    ax.axvline(results["config"]["degrade_at_round"], color=style.GRID, linewidth=1.0)
    ax.text(
        results["config"]["degrade_at_round"] + 0.15,
        0.885,
        "judge silently degrades\n(0.85 → 0.72)",
        color=style.INK_2,
        fontsize=8.5,
    )
    alarm = results["alarm_round"]
    if alarm is not None:
        y = rounds[alarm]["wilson_high"]
        ax.plot([alarm], [y], marker="o", markersize=7, color=style.SERIES_1)
        ax.annotate(
            f"sensor alarm @ round {alarm}",
            (alarm, y),
            textcoords="offset points",
            xytext=(10, 10),
            color=style.INK,
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("Calibration round (frozen anchor set, re-judged)")
    ax.set_ylabel("Judge accuracy vs anchors")
    ax.set_xticks(range(0, N_ROUNDS, 2))
    style.titles(
        ax,
        "The monitoring system alarms on its own failing sensor",
        f"{N_ANCHORS} frozen anchor pairs re-judged per round · alarm rule: CI upper bound "
        f"below the {TOLERANCE_FLOOR:.0%} tolerance floor",
    )
    ax.legend(frameon=False, loc="lower left", labelcolor=style.INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "judge_calibration.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = asyncio.run(run(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "judge_calibration.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
