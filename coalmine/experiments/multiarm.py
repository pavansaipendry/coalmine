"""Phase 4 experiment: k challengers without alpha spending is peeking again.

Four challengers shadow-test against the champion, each watched by an
always-valid promotion test. Uncorrected (each at α = 5%), the chance of
promoting SOME not-better arm compounds across arms — the multi-arm version
of the peeking pathology this project exists to kill. Splitting the budget
(α/4 per arm) restores family-wise control by union bound, and the price is
measured: the truly better arm takes longer to promote.

Run:  python -m coalmine.experiments.multiarm
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.experiments import style
from coalmine.sim.runner import run_msprt_batch
from coalmine.sim.verdicts import gen_wins

K = 4
ALPHA = 0.05
HORIZON = 20_000
NULL_TRIALS = 2_000
GOOD_ARM_W = 0.55
LATENCY_TRIALS = 1_500


def run(seed: int) -> dict:
    rng = np.random.default_rng(seed)

    print(f"  family-wise error under the global null ({K} arms, {NULL_TRIALS} trials) ...")
    rejected_uncorr = np.zeros(NULL_TRIALS, dtype=bool)
    rejected_corr = np.zeros(NULL_TRIALS, dtype=bool)
    for _ in range(K):
        wins = gen_wins(rng, NULL_TRIALS, HORIZON, 0.5)
        rejected_uncorr |= run_msprt_batch(wins, 0.5, ALPHA) > 0
        rejected_corr |= run_msprt_batch(wins, 0.5, ALPHA / K) > 0
    fwer_uncorr = float(rejected_uncorr.mean())
    fwer_corr = float(rejected_corr.mean())
    print(f"  FWER uncorrected {fwer_uncorr:.1%} vs corrected {fwer_corr:.1%}")

    print(f"  promotion latency for a truly better arm (w = {GOOD_ARM_W}) ...")
    good = gen_wins(rng, LATENCY_TRIALS, HORIZON, GOOD_ARM_W)
    t_uncorr = run_msprt_batch(good, 0.5, ALPHA)
    t_corr = run_msprt_batch(good, 0.5, ALPHA / K)
    lat_uncorr = float(np.median(t_uncorr[t_uncorr > 0]))
    lat_corr = float(np.median(t_corr[t_corr > 0]))
    print(
        f"  median promotion: {lat_uncorr:.0f} verdicts at α, {lat_corr:.0f} at α/{K} "
        f"(+{lat_corr / lat_uncorr - 1:.0%})"
    )

    return {
        "config": {
            "k": K,
            "alpha": ALPHA,
            "horizon": HORIZON,
            "null_trials": NULL_TRIALS,
            "good_arm_w": GOOD_ARM_W,
            "latency_trials": LATENCY_TRIALS,
            "seed": seed,
        },
        "fwer_uncorrected": fwer_uncorr,
        "fwer_corrected": fwer_corr,
        "latency_uncorrected": lat_uncorr,
        "latency_corrected": lat_corr,
        "latency_cost": lat_corr / lat_uncorr - 1,
    }


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax in (ax1, ax2):
        style.style_axes(ax)

    labels = [f"Per-arm α = {ALPHA:.0%}\n(uncorrected)", f"Per-arm α = {ALPHA / K:.2%}\n(α / k)"]
    fwer = [results["fwer_uncorrected"], results["fwer_corrected"]]
    bars = ax1.bar(labels, fwer, width=0.45, color=[style.SERIES_1, style.SERIES_2])
    for bar, v in zip(bars, fwer):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.1%}",
            ha="center", color=style.INK, fontsize=11, fontweight="bold",
        )
    ax1.axhline(ALPHA, color=style.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax1.text(1.42, ALPHA + 0.003, "5% budget", color=style.MUTED, fontsize=8.5, ha="right")
    ax1.set_ylabel("P(any false promotion across 4 arms)")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style.titles(
        ax1,
        "Four arms compound the error",
        f"{K} null challengers, {results['config']['null_trials']:,} trials",
    )

    lat = [results["latency_uncorrected"], results["latency_corrected"]]
    bars = ax2.bar(labels, lat, width=0.45, color=[style.SERIES_1, style.SERIES_2])
    for bar, v in zip(bars, lat):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, v + 12, f"{v:.0f}",
            ha="center", color=style.INK, fontsize=11, fontweight="bold",
        )
    ax2.set_ylabel("Median verdicts to promote the good arm")
    style.titles(
        ax2,
        "The price of protection",
        f"Truly better arm (w = {GOOD_ARM_W}): +{results['latency_cost']:.0%} latency",
    )
    fig.tight_layout(w_pad=3.0)
    fig.savefig(out_dir / "multiarm.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = run(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "multiarm.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
