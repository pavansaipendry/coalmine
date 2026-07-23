"""Headline experiment: detection latency vs regression size at a controlled false-alarm rate.

A challenger serves at true-parity (50% win rate) until an injected regression
drops it by a known amount at a known changepoint. Verdicts arrive through the
judge channel (accuracy 0.85, so every true shift reaches the detector
attenuated to 70% of its size). A CUSUM tuned to p1 = 0.45 on the observed
scale, with its threshold calibrated to a 5% false-alarm budget over the
monitoring horizon, watches the stream.

Also measures the motivating baseline on the same null streams: a repeated
two-sided t-test peeking every 100 verdicts, which claims alpha = 5% per check
but false-alarms far more often over the horizon.

Run:  python -m coalmine.experiments.detection_latency
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.experiments import style
from coalmine.sim.runner import (
    calibrate_cusum_threshold,
    naive_peeking_false_alarm_rate,
    run_cusum_batch,
)
from coalmine.sim.verdicts import gen_wins, gen_wins_changepoint
from coalmine.stats.channel import JudgeChannel

HORIZON = 20_000  # decisive verdicts monitored per trial
CHANGEPOINT = 5_000
P0 = 0.50
P1_DESIGN = 0.45  # observed-scale shift the CUSUM is tuned to detect
FALSE_ALARM_BUDGET = 0.05  # over the full horizon
TRUE_SHIFTS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
CALIB_TRIALS = 4_000
NULL_TRIALS = 2_000
TRIALS_PER_SHIFT = 1_500
SAMPLING_RATE = 0.20  # fraction of live requests judged


def run(seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    channel = JudgeChannel(accuracy=0.85, tie_rate=0.10)
    rpv = channel.requests_per_verdict(SAMPLING_RATE)

    print(f"calibrating CUSUM threshold on {CALIB_TRIALS} null trials ...")
    calib = gen_wins(rng, CALIB_TRIALS, HORIZON, P0)
    h = calibrate_cusum_threshold(calib, P0, P1_DESIGN, FALSE_ALARM_BUDGET)

    print(f"measuring realized false-alarm rates on {NULL_TRIALS} fresh null trials ...")
    null_wins = gen_wins(rng, NULL_TRIALS, HORIZON, P0)
    cusum_fpr = float((run_cusum_batch(null_wins, P0, P1_DESIGN, h) > 0).mean())
    naive_fpr = naive_peeking_false_alarm_rate(null_wins, check_every=100, min_n=100, alpha=0.05)

    per_shift = []
    for shift in TRUE_SHIFTS:
        observed_after = P0 - shift * channel.attenuation()
        wins = gen_wins_changepoint(
            rng, TRIALS_PER_SHIFT, HORIZON, P0, observed_after, CHANGEPOINT
        )
        alarms = run_cusum_batch(wins, P0, P1_DESIGN, h)
        premature = int(((alarms > 0) & (alarms <= CHANGEPOINT)).sum())
        caught = alarms[alarms > CHANGEPOINT] - CHANGEPOINT
        detect_rate = len(caught) / TRIALS_PER_SHIFT
        q25, med, q75 = (np.percentile(caught, [25, 50, 75]) if len(caught) else [np.nan] * 3)
        per_shift.append(
            {
                "true_shift": shift,
                "observed_shift": shift * channel.attenuation(),
                "detect_rate": detect_rate,
                "premature_alarms": premature,
                "latency_verdicts_q25": float(q25),
                "latency_verdicts_median": float(med),
                "latency_verdicts_q75": float(q75),
                "latency_requests_median": float(med * rpv) if np.isfinite(med) else None,
            }
        )
        print(
            f"  true shift {shift:>5.3f}: caught {detect_rate:5.1%}, "
            f"median latency {med:7.0f} verdicts (~{med * rpv:8.0f} requests)"
        )

    results = {
        "config": {
            "horizon": HORIZON,
            "changepoint": CHANGEPOINT,
            "p0": P0,
            "p1_design": P1_DESIGN,
            "false_alarm_budget": FALSE_ALARM_BUDGET,
            "judge_accuracy": channel.accuracy,
            "tie_rate": channel.tie_rate,
            "sampling_rate": SAMPLING_RATE,
            "requests_per_verdict": rpv,
            "seed": seed,
            "trials_per_shift": TRIALS_PER_SHIFT,
        },
        "threshold_h": h,
        "cusum_false_alarm_rate": cusum_fpr,
        "naive_peeking_false_alarm_rate": naive_fpr,
        "per_shift": per_shift,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detection_latency.json").write_text(json.dumps(results, indent=2))
    print(
        f"\nfalse alarms over {HORIZON} verdicts: "
        f"CUSUM {cusum_fpr:.1%} (budget {FALSE_ALARM_BUDGET:.0%}) vs "
        f"naive peeking {naive_fpr:.1%}"
    )
    return results


def chart_latency(results: dict, out_dir: Path) -> None:
    style.setup()
    per_shift = [r for r in results["per_shift"] if np.isfinite(r["latency_verdicts_median"])]
    x = [r["true_shift"] * 100 for r in per_shift]
    med = [r["latency_verdicts_median"] for r in per_shift]
    q25 = [r["latency_verdicts_q25"] for r in per_shift]
    q75 = [r["latency_verdicts_q75"] for r in per_shift]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    style.style_axes(ax)
    ax.fill_between(x, q25, q75, color=style.BAND_1, alpha=0.55, linewidth=0, label="IQR")
    ax.plot(x, med, color=style.SERIES_1, linewidth=1.8, marker="o", markersize=6, label="Median")
    for r in per_shift:
        if r["detect_rate"] < 0.995:
            ax.annotate(
                f"caught {r['detect_rate']:.0%}",
                (r["true_shift"] * 100, r["latency_verdicts_median"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8.5,
                color=style.INK_2,
            )
    ax.set_yscale("log")
    ax.set_xlabel("True win-rate regression (percentage points)")
    ax.set_ylabel("Detection latency (decisive verdicts, log)")
    cfg = results["config"]
    style.titles(
        ax,
        "Smaller regressions take quadratically longer to catch",
        f"CUSUM calibrated to a 5% false-alarm budget over {cfg['horizon']:,} verdicts "
        f"(realized: {results['cusum_false_alarm_rate']:.1%})\n"
        f"Judge accuracy {cfg['judge_accuracy']:.0%} attenuates true shifts to "
        f"{2 * cfg['judge_accuracy'] - 1:.0%} observed  ·  1 verdict ≈ "
        f"{cfg['requests_per_verdict']:.1f} live requests at {cfg['sampling_rate']:.0%} sampling",
    )
    ax.legend(frameon=False, loc="upper right", labelcolor=style.INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "detection_latency.png")
    plt.close(fig)


def chart_false_alarms(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    style.style_axes(ax)
    methods = ["Naive repeated t-test\n(peek every 100 verdicts)", "CUSUM\n(calibrated threshold)"]
    rates = [results["naive_peeking_false_alarm_rate"], results["cusum_false_alarm_rate"]]
    colors = [style.SERIES_1, style.SERIES_2]
    bars = ax.bar(methods, rates, width=0.45, color=colors)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.012,
            f"{rate:.1%}",
            ha="center",
            color=style.INK,
            fontsize=11,
            fontweight="bold",
        )
    ax.axhline(0.05, color=style.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(
        1.38, 0.053, "5% budget", color=style.MUTED, fontsize=8.5, ha="right", va="bottom"
    )
    ax.set_ylabel("P(at least one false alarm over the horizon)")
    ax.set_ylim(0, max(rates) * 1.25)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cfg = results["config"]
    style.titles(
        ax,
        "Peeking inflates false alarms; the sequential test holds its budget",
        f"Same {results['config']['seed']}-seeded null streams, {cfg['horizon']:,} verdicts, "
        "both methods nominally at 5%",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "false_alarm_comparison.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = run(args.seed, args.out)
    chart_latency(results, args.out)
    chart_false_alarms(results, args.out)
    print(f"charts written to {args.out}/")


if __name__ == "__main__":
    main()
