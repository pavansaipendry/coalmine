"""Implementation validation: empirical SPRT behavior vs Wald's closed-form theory.

Runs the promotion-gate SPRT (H0: p = 0.50 vs H1: p = 0.55, alpha 5%, beta 10%)
across a grid of actual win rates and compares two empirical curves against
Wald's approximations: the probability of promoting, and the expected verdicts
to a decision. Matching theory is the evidence the implementation is correct;
the known small gap (empirical E[N] slightly above theory) is boundary
overshoot, which Wald's approximation ignores.

Run:  python -m coalmine.experiments.sprt_validation
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.experiments import style
from coalmine.sim.runner import run_sprt_batch
from coalmine.sim.verdicts import gen_wins
from coalmine.stats.wald import accept_h1_probability, expected_n

P0, P1 = 0.50, 0.55
ALPHA, BETA = 0.05, 0.10
GRID = np.round(np.arange(0.40, 0.6401, 0.02), 3)
TRIALS = 2_000
MAX_N = 20_000


def run(seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for p in GRID:
        wins = gen_wins(rng, TRIALS, MAX_N, p)
        decisions, stops = run_sprt_batch(wins, P0, P1, ALPHA, BETA)
        censored = int((decisions == 0).sum())
        rows.append(
            {
                "p": float(p),
                "promote_rate": float((decisions == 1).mean()),
                "mean_n": float(stops.mean()),
                "censored": censored,
                "theory_promote": accept_h1_probability(p, P0, P1, ALPHA, BETA),
                "theory_n": expected_n(p, P0, P1, ALPHA, BETA),
            }
        )
        print(
            f"  p={p:.2f}: promote {rows[-1]['promote_rate']:6.1%} "
            f"(theory {rows[-1]['theory_promote']:6.1%}), "
            f"E[N] {rows[-1]['mean_n']:7.0f} (theory {rows[-1]['theory_n']:7.0f})"
            + (f", censored {censored}" if censored else "")
        )

    results = {
        "config": {
            "p0": P0,
            "p1": P1,
            "alpha": ALPHA,
            "beta": BETA,
            "trials": TRIALS,
            "max_n": MAX_N,
            "seed": seed,
        },
        "grid": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sprt_validation.json").write_text(json.dumps(results, indent=2))
    return results


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    rows = results["grid"]
    p = [r["p"] for r in rows]
    dense = np.linspace(min(p), max(p), 200)
    theory_promote = [accept_h1_probability(x, P0, P1, ALPHA, BETA) for x in dense]
    theory_n = [expected_n(x, P0, P1, ALPHA, BETA) for x in dense]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax in (ax1, ax2):
        style.style_axes(ax)
        ax.set_xlabel("Actual challenger win rate (observed scale)")

    ax1.plot(dense, theory_promote, color=style.SERIES_1, linewidth=1.8, label="Wald theory")
    ax1.plot(
        p,
        [r["promote_rate"] for r in rows],
        color=style.SERIES_2,
        linestyle="none",
        marker="o",
        markersize=6,
        label="Simulation",
    )
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax1.set_ylabel("P(promote challenger)")
    style.titles(
        ax1,
        "Operating characteristic",
        f"α = {ALPHA:.0%} at p₀ = {P0}, power = {1 - BETA:.0%} at p₁ = {P1}",
    )
    ax1.legend(frameon=False, loc="upper left", labelcolor=style.INK_2, fontsize=9)

    ax2.plot(dense, theory_n, color=style.SERIES_1, linewidth=1.8, label="Wald theory")
    ax2.plot(
        p,
        [r["mean_n"] for r in rows],
        color=style.SERIES_2,
        linestyle="none",
        marker="o",
        markersize=6,
        label="Simulation",
    )
    ax2.set_ylabel("Expected verdicts to a decision")
    style.titles(
        ax2,
        "Decision cost peaks at the boundary",
        f"{results['config']['trials']:,} trials per point; simulation sits\n"
        "slightly above theory (boundary overshoot)",
    )
    ax2.legend(frameon=False, loc="upper right", labelcolor=style.INK_2, fontsize=9)

    fig.tight_layout(w_pad=3.0)
    fig.savefig(out_dir / "sprt_validation.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = run(args.seed, args.out)
    chart(results, args.out)
    print(f"chart written to {args.out}/")


if __name__ == "__main__":
    main()
