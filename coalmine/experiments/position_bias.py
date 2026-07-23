"""Phase 3 experiment: position bias — measured, and neutralized by randomization.

One equal-quality traffic run (ε = 0, both configs serve the good pool) is
judged twice by the same judge with a deliberate 20% first-position
preference: once with randomized presentation, once always showing the
champion first. Randomized judging keeps the challenger's win rate at parity
and exposes the bias in position space; fixed-order judging silently corrupts
the win-rate estimate by ~10 points — which is why every real comparison in
this system randomizes.

Run:  python -m coalmine.experiments.position_bias
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from coalmine.core.store import SQLiteEventStore
from coalmine.experiments import style
from coalmine.judging.judges import NoisyOracleJudge
from coalmine.judging.metrics import challenger_win_rate, measured_position_bias
from coalmine.judging.pipeline import JudgePipeline
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario, run_simulation

N_REQUESTS = 4_000
POSITION_BIAS = 0.20
JUDGE = dict(accuracy=0.85, tie_rate=0.10, position_bias=POSITION_BIAS)


async def run(seed: int) -> dict:
    queries = synthetic_queries(400)
    pool = ResponsePool.build_synthetic(queries)
    store = SQLiteEventStore(Path(tempfile.mkdtemp()) / "position_bias.db")
    scenario = Scenario(
        run_id="bias-traffic", seed=seed, n_requests=N_REQUESTS, shadow_rate=1.0
    )
    await run_simulation(scenario, queries, pool, store)

    judge = NoisyOracleJudge(**JUDGE)
    randomized = await JudgePipeline(judge, seed=seed, judge_id="randomized").judge_run(
        store, "bias-traffic", randomize=True
    )
    fixed = await JudgePipeline(judge, seed=seed, judge_id="fixed-order").judge_run(
        store, "bias-traffic", randomize=False
    )
    store.close()

    results = {
        "config": {"n_requests": N_REQUESTS, "judge": JUDGE, "seed": seed},
        "true_challenger_win_rate": 0.5,
        "randomized_win_rate": challenger_win_rate(randomized),
        "fixed_order_win_rate": challenger_win_rate(fixed),
        "measured_position_bias": measured_position_bias(randomized),
    }
    print(
        f"  true win rate 0.500 · randomized order: {results['randomized_win_rate']:.3f} · "
        f"champion-always-first: {results['fixed_order_win_rate']:.3f}"
    )
    print(
        f"  measured position bias (randomized pass): "
        f"{results['measured_position_bias']:+.1%} toward first-shown"
    )
    return results


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    style.style_axes(ax)
    labels = ["Randomized\npresentation", "Champion always\nshown first"]
    values = [results["randomized_win_rate"], results["fixed_order_win_rate"]]
    bars = ax.bar(labels, values, width=0.45, color=[style.SERIES_1, style.SERIES_2])
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.012,
            f"{v:.1%}",
            ha="center",
            color=style.INK,
            fontsize=11,
            fontweight="bold",
        )
    ax.axhline(0.5, color=style.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(1.42, 0.503, "true parity", color=style.MUTED, fontsize=8.5, ha="right")
    ax.set_ylabel("Measured challenger win rate")
    ax.set_ylim(0, 0.62)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style.titles(
        ax,
        "Position bias corrupts unrandomized judging",
        f"Same equal-quality responses, same judge with a {POSITION_BIAS:.0%} first-position "
        f"preference\nmeasured bias {results['measured_position_bias']:+.1%} toward "
        "first-shown — randomization cancels it; fixed order does not",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "position_bias.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = asyncio.run(run(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "position_bias.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
