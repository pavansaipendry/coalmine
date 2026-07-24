"""Ensemble-accelerated detection: buy back the judge channel's attenuation.

Detection latency scales like 1/shift² and the judge channel attenuates every
true shift by (2a_eff − 1) — so raising effective judge accuracy pays off
quadratically. A majority-of-k ensemble does exactly that (no-bias reference:
majority_accuracy(0.85, 3) = 0.939). This experiment measures the whole
chain: effective accuracy and tie rate of simulated ensembles (including the
15% position bias and randomized presentation used everywhere else), then
vectorized CUSUM detection latency at k = 1, 3, 5 across regression sizes —
the latency-vs-judge-cost tradeoff curve, in requests.

The null is untouched by k (equal-quality pairs stay at 50% by symmetry,
measured below), so the false-alarm budget carries over unchanged.

Run:  python -m coalmine.experiments.ensemble_detection
"""

import argparse
import asyncio
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.experiments import style
from coalmine.judging.ensemble import EnsembleJudge, majority_accuracy
from coalmine.judging.judges import NoisyOracleJudge, compare_randomized
from coalmine.sim.runner import calibrate_cusum_threshold, run_cusum_batch
from coalmine.sim.verdicts import gen_wins_changepoint

MEMBER = dict(accuracy=0.85, tie_rate=0.10, position_bias=0.15)
KS = [1, 3, 5]
TRUE_SHIFTS = [0.03, 0.05]
SAMPLING_RATE = 0.25
CALIBRATION_COMPARISONS = 20_000
PREFIX = 5_000
HORIZON = 20_000
TRIALS = 1_500
P0, P1_DESIGN = 0.50, 0.45
GOOD = {"text": "grounded", "source_pool": "good"}
BAD = {"text": "vague", "source_pool": "bad"}


def _judge(k: int, seed_tag: int):
    if k == 1:
        return NoisyOracleJudge(**MEMBER, judge_id=f"solo-{seed_tag}")
    members = [
        NoisyOracleJudge(**MEMBER, judge_id=f"m{seed_tag}-{i}") for i in range(k)
    ]
    return EnsembleJudge(members, judge_id=f"ensemble{k}-{seed_tag}")


async def _measure_channel(k: int, seed: int) -> dict:
    """Empirical effective accuracy / tie rate of a k-ensemble on unequal
    pairs, and the null win rate on equal pairs (symmetry check)."""
    judge = _judge(k, seed)
    correct = decisive = 0
    for i in range(CALIBRATION_COMPARISONS):
        result = await compare_randomized(judge, "q", GOOD, BAD, np.random.default_rng([seed, i]))
        if result.winner != "tie":
            decisive += 1
            correct += result.winner == "a"
    equal_judge = _judge(k, seed + 1)
    b_wins = eq_decisive = 0
    for i in range(CALIBRATION_COMPARISONS // 2):
        result = await compare_randomized(
            equal_judge, "q", GOOD, GOOD, np.random.default_rng([seed + 1, i])
        )
        if result.winner != "tie":
            eq_decisive += 1
            b_wins += result.winner == "b"
    return {
        "k": k,
        "effective_accuracy": correct / decisive,
        "decisive_rate": decisive / CALIBRATION_COMPARISONS,
        "null_win_rate": b_wins / eq_decisive,
    }


def run(seed: int) -> dict:
    channels = [
        asyncio.run(_measure_channel(k, seed + 10 * k)) for k in KS
    ]
    for ch in channels:
        print(
            f"  k={ch['k']}: effective accuracy {ch['effective_accuracy']:.3f} "
            f"(no-bias reference {majority_accuracy(MEMBER['accuracy'], ch['k']):.3f}), "
            f"decisive rate {ch['decisive_rate']:.3f}, "
            f"null win rate {ch['null_win_rate']:.3f}"
        )

    rng = np.random.default_rng(seed)
    calib = gen_wins_changepoint(rng, 4_000, HORIZON, P0, P0, 0)
    h = calibrate_cusum_threshold(calib, P0, P1_DESIGN, 0.05)

    rows = []
    for ch in channels:
        a_eff = ch["effective_accuracy"]
        attenuation = 2 * a_eff - 1
        requests_per_verdict = 1.0 / (SAMPLING_RATE * ch["decisive_rate"])
        for shift in TRUE_SHIFTS:
            w_after = P0 - shift * attenuation
            wins = gen_wins_changepoint(rng, TRIALS, HORIZON, P0, w_after, PREFIX)
            alarms = run_cusum_batch(wins, P0, P1_DESIGN, h)
            caught = alarms[alarms > PREFIX] - PREFIX
            rows.append(
                {
                    "k": ch["k"],
                    "true_shift": shift,
                    "observed_shift": shift * attenuation,
                    "detect_rate": len(caught) / TRIALS,
                    "median_latency_verdicts": float(np.median(caught)),
                    "median_latency_requests": float(np.median(caught) * requests_per_verdict),
                    "judge_calls_per_pair": ch["k"],
                }
            )
            print(
                f"  k={ch['k']} shift={shift:.2f}: median latency "
                f"{rows[-1]['median_latency_requests']:8.0f} requests "
                f"({rows[-1]['detect_rate']:5.1%} caught)"
            )

    return {
        "config": {
            "member": MEMBER,
            "ks": KS,
            "true_shifts": TRUE_SHIFTS,
            "sampling_rate": SAMPLING_RATE,
            "prefix": PREFIX,
            "horizon": HORIZON,
            "trials": TRIALS,
            "seed": seed,
        },
        "channels": channels,
        "rows": rows,
    }


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    style.style_axes(ax)
    rows = results["rows"]
    x = np.arange(len(KS))
    width = 0.34
    colors = {TRUE_SHIFTS[0]: style.SERIES_1, TRUE_SHIFTS[1]: style.SERIES_2}
    for j, shift in enumerate(TRUE_SHIFTS):
        vals = [
            next(
                r["median_latency_requests"]
                for r in rows
                if r["k"] == k and r["true_shift"] == shift
            )
            for k in KS
        ]
        offset = (j - 0.5) * (width + 0.04)
        bars = ax.bar(
            x + offset,
            vals,
            width,
            color=colors[shift],
            label=f"{shift:.0%} true regression",
        )
        for bar, v, k in zip(bars, vals, KS):
            base = vals[0]
            note = f"{v:,.0f}" if k == 1 else f"{v:,.0f}\n(−{1 - v / base:.0%})"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 120,
                note,
                ha="center",
                color=style.INK,
                fontsize=9,
            )
    channels = {c["k"]: c for c in results["channels"]}
    ax.set_xticks(
        x,
        [
            f"k = {k}\nacc {channels[k]['effective_accuracy']:.2f} · {k}× cost"
            for k in KS
        ],
    )
    ax.set_ylabel("Median detection latency (requests)")
    style.titles(
        ax,
        "Majority-of-k judging detects regressions sooner",
        "Higher effective accuracy shrinks the channel attenuation — a quadratic win\n"
        "same members (0.85 acc, 10% ties, 15% position bias) · same 5% false-alarm "
        "budget · null untouched by k",
    )
    ax.legend(frameon=False, loc="upper right", labelcolor=style.INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ensemble_detection.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=139)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = run(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ensemble_detection.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
