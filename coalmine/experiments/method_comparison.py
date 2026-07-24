"""Phase 4 headline: three sequential methods, compared where each one lives.

Decision problem (iid stream, "is the challenger better?"): Wald's SPRT
(point alternative, terminates either way), the one-sided mixture SPRT
(always-valid, no alternative to tune), and the stitched confidence sequence
(always-valid, closed form, no prior) — median time-to-promotion across true
effect sizes, plus realized false-promotion rates on null streams. All three
run at a nominal 5%; the Monte Carlo measures what each actually delivers.

Changepoint problem (regression starting mid-stream): the anytime methods
test a FIXED hypothesis over the whole stream, so a healthy prefix dilutes
their evidence — detection latency grows with prefix length, while CUSUM
(which resets its statistic at zero) stays flat. This is why the promotion
gate and the regression alarm are different primitives.

Run:  python -m coalmine.experiments.method_comparison   (takes a few minutes)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.experiments import style
from coalmine.sim.runner import (
    calibrate_cusum_threshold,
    run_cusum_batch,
    run_msprt_batch,
    run_sprt_batch,
    run_stitched_batch,
)
from coalmine.sim.verdicts import gen_wins, gen_wins_changepoint

P0 = 0.50
SPRT_P1 = 0.55
ALPHA = 0.05
HORIZON = 20_000
EFFECT_GRID = [0.52, 0.54, 0.55, 0.56, 0.58, 0.60]
TRIALS = 1_500
NULL_TRIALS = 4_000

PREFIXES = [0, 2_500, 5_000, 10_000]
POST_WINDOW = 15_000
DROP_TO = 0.45
DILUTION_TRIALS = 800
CUSUM_CALIB_TRIALS = 4_000

METHOD_COLORS = {"SPRT": "#2a78d6", "mSPRT": "#008300", "Stitched CS": "#e87ba4"}


def _summary(times: np.ndarray, trials: int) -> dict:
    detected = times[times > 0]
    return {
        "detect_rate": len(detected) / trials,
        "median_time": float(np.median(detected)) if len(detected) else None,
    }


def run_decision(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for w in EFFECT_GRID:
        wins = gen_wins(rng, TRIALS, HORIZON, w)
        dec, stops = run_sprt_batch(wins, P0, SPRT_P1, ALPHA, beta=0.10)
        sprt_times = np.where(dec == 1, stops, -1)
        row = {
            "w": w,
            "SPRT": _summary(sprt_times, TRIALS),
            "mSPRT": _summary(run_msprt_batch(wins, P0, ALPHA), TRIALS),
            "Stitched CS": _summary(run_stitched_batch(wins, P0, ALPHA), TRIALS),
        }
        rows.append(row)
        print(
            f"  w={w:.2f}: " + "  ".join(
                f"{m} {row[m]['median_time'] or float('nan'):6.0f} "
                f"({row[m]['detect_rate']:5.1%})"
                for m in METHOD_COLORS
            )
        )

    null = gen_wins(rng, NULL_TRIALS, HORIZON, P0)
    dec_null, _ = run_sprt_batch(null, P0, SPRT_P1, ALPHA, beta=0.10)
    fpr = {
        "SPRT": float((dec_null == 1).mean()),
        "mSPRT": float((run_msprt_batch(null, P0, ALPHA) > 0).mean()),
        "Stitched CS": float((run_stitched_batch(null, P0, ALPHA) > 0).mean()),
    }
    print("  null false-promotion: " + "  ".join(f"{m} {v:.2%}" for m, v in fpr.items()))
    return {"grid": rows, "false_promotion": fpr}


def run_dilution(seed: int) -> dict:
    rng = np.random.default_rng(seed + 1)
    rows = []
    for prefix in PREFIXES:
        horizon = prefix + POST_WINDOW
        calib = gen_wins(rng, CUSUM_CALIB_TRIALS, horizon, P0)
        h = calibrate_cusum_threshold(calib, P0, DROP_TO, ALPHA)
        wins = gen_wins_changepoint(rng, DILUTION_TRIALS, horizon, P0, DROP_TO, prefix)

        cusum_alarms = run_cusum_batch(wins, P0, DROP_TO, h)
        cusum_times = np.where(cusum_alarms > prefix, cusum_alarms, -1)
        # The anytime methods detect the drop as champion improvement on the
        # complement stream — same one-sided machinery, mirrored.
        complement = ~wins
        msprt_raw = run_msprt_batch(complement, P0, ALPHA)
        cs_raw = run_stitched_batch(complement, P0, ALPHA)
        msprt_times = np.where(msprt_raw > prefix, msprt_raw, -1)
        cs_times = np.where(cs_raw > prefix, cs_raw, -1)

        def latency(times: np.ndarray) -> dict:
            s = _summary(times, DILUTION_TRIALS)
            if s["median_time"] is not None:
                s["median_latency"] = s["median_time"] - prefix
            else:
                s["median_latency"] = None
            return s

        row = {
            "prefix": prefix,
            "CUSUM": latency(cusum_times),
            "mSPRT": latency(msprt_times),
            "Stitched CS": latency(cs_times),
        }
        rows.append(row)
        print(
            f"  prefix {prefix:6,}: " + "  ".join(
                f"{m} {row[m]['median_latency'] or float('nan'):6.0f}"
                for m in ("CUSUM", "mSPRT", "Stitched CS")
            )
        )
    return {"rows": rows}


def chart_decision(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.6), width_ratios=[1.5, 1])
    for ax in (ax1, ax2):
        style.style_axes(ax)

    grid = results["decision"]["grid"]
    x = [r["w"] for r in grid]
    label_offsets = {"SPRT": (8, 8), "mSPRT": (8, -12), "Stitched CS": (8, -3)}
    for method, color in METHOD_COLORS.items():
        y = [r[method]["median_time"] for r in grid]
        ax1.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=5, label=method)
        ax1.annotate(
            method,
            (x[-1], y[-1]),
            textcoords="offset points",
            xytext=label_offsets[method],
            color=style.INK_2,
            fontsize=8.5,
        )
    ax1.set_yscale("log")
    ax1.set_xlabel("True challenger win rate (observed scale)")
    ax1.set_ylabel("Median verdicts to promotion (log)")
    style.titles(
        ax1,
        "Anytime validity has a price — and a prior helps pay it",
        "Median time to promote a truly better challenger; all methods nominal α = 5%",
    )
    ax1.legend(frameon=False, loc="upper right", labelcolor=style.INK_2, fontsize=9)

    fpr = results["decision"]["false_promotion"]
    methods = list(METHOD_COLORS)
    bars = ax2.bar(
        range(len(methods)),
        [fpr[m] for m in methods],
        width=0.5,
        color=[METHOD_COLORS[m] for m in methods],
    )
    for bar, m in zip(bars, methods):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            fpr[m] + 0.002,
            f"{fpr[m]:.1%}",
            ha="center",
            color=style.INK,
            fontsize=10,
            fontweight="bold",
        )
    ax2.axhline(ALPHA, color=style.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax2.text(2.35, ALPHA + 0.002, "5% budget", color=style.MUTED, fontsize=8.5, ha="right")
    ax2.set_xticks(range(len(methods)), methods)
    ax2.set_ylabel("False promotions (null streams)")
    ax2.set_ylim(0, 0.075)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style.titles(
        ax2, "All hold the error budget", f"{NULL_TRIALS:,} null runs, {HORIZON:,} verdicts"
    )
    fig.tight_layout(w_pad=3.0)
    fig.savefig(out_dir / "method_comparison.png")
    plt.close(fig)


def chart_dilution(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    style.style_axes(ax)
    rows = results["dilution"]["rows"]
    x = [r["prefix"] for r in rows]
    colors = {"CUSUM": "#2a78d6", "mSPRT": "#008300", "Stitched CS": "#e87ba4"}
    for method, color in colors.items():
        y = [r[method]["median_latency"] for r in rows]
        ax.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=5, label=method)
        ax.annotate(
            method,
            (x[-1], y[-1]),
            textcoords="offset points",
            xytext=(8, -3),
            color=style.INK_2,
            fontsize=8.5,
        )
    ax.set_xlabel("Healthy verdicts before the regression begins")
    ax.set_ylabel("Median detection latency (verdicts)")
    style.titles(
        ax,
        "Fixed-hypothesis tests dilute; the changepoint detector does not",
        f"Win rate drops {P0:.2f} → {DROP_TO:.2f} after a healthy prefix\n"
        "Anytime methods test the whole stream, so the prefix buries the shift · "
        "CUSUM restarts its evidence at zero",
    )
    ax.legend(frameon=False, loc="upper left", labelcolor=style.INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "changepoint_dilution.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    print("decision problem:")
    decision = run_decision(args.seed)
    print("changepoint dilution:")
    dilution = run_dilution(args.seed)
    results = {
        "config": {
            "p0": P0,
            "sprt_p1": SPRT_P1,
            "alpha": ALPHA,
            "horizon": HORIZON,
            "trials": TRIALS,
            "null_trials": NULL_TRIALS,
            "prefixes": PREFIXES,
            "drop_to": DROP_TO,
            "seed": args.seed,
        },
        "decision": decision,
        "dilution": dilution,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "method_comparison.json").write_text(json.dumps(results, indent=2))
    chart_decision(results, args.out)
    chart_dilution(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
