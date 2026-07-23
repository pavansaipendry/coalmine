"""Phase 2 experiment: does shadowing cost the user anything?

Three paced simulation runs share one seed, so arrival times and champion
latency draws are IDENTICAL — the only difference is the shadow rate (0%,
20%, 100%). Any user-facing latency difference is therefore attributable to
shadow overhead, not noise. The challenger is deliberately slower than the
champion (120ms vs 80ms median): if shadow calls leaked into the user path,
these percentiles would show it immediately.

Run:  python -m coalmine.experiments.shadow_overhead
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.core.store import SQLiteEventStore
from coalmine.experiments import style
from coalmine.serving.backends import LogNormalLatency
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario, run_simulation

SHADOW_RATES = [0.0, 0.2, 1.0]
N_REQUESTS = 1_200
RATE_RPS = 60.0
PERCENTILES = [50, 95, 99]
SERIES_COLORS = ["#2a78d6", "#008300", "#e87ba4"]  # categorical slots 1-3


async def run(seed: int) -> dict:
    queries = synthetic_queries(400)
    pool = ResponsePool.build_synthetic(queries)
    store = SQLiteEventStore(Path(tempfile.mkdtemp()) / "shadow_overhead.db")
    rows = []
    for rate in SHADOW_RATES:
        scenario = Scenario(
            run_id=f"shadow-{int(rate * 100)}",
            seed=seed,
            n_requests=N_REQUESTS,
            rate_rps=RATE_RPS,
            shadow_rate=rate,
            champion_latency=LogNormalLatency(80.0, 0.4),
            challenger_latency=LogNormalLatency(120.0, 0.5),
        )
        result = await run_simulation(scenario, queries, pool, store)
        pcts = {f"p{p}": float(np.percentile(result.user_latency_ms, p)) for p in PERCENTILES}
        rows.append(
            {
                "shadow_rate": rate,
                "shadow_responses": result.counts.get("shadow_response", 0),
                **pcts,
            }
        )
        print(
            f"  shadow {rate:4.0%}: "
            + "  ".join(f"{k} {v:7.1f}ms" for k, v in pcts.items())
            + f"  ({rows[-1]['shadow_responses']} shadow calls)"
        )
    store.close()

    baseline = rows[0]
    for row in rows:
        row["p99_delta_vs_off"] = row["p99"] - baseline["p99"]
    return {
        "config": {
            "n_requests": N_REQUESTS,
            "rate_rps": RATE_RPS,
            "seed": seed,
            "champion_latency": {"median_ms": 80.0, "sigma": 0.4},
            "challenger_latency": {"median_ms": 120.0, "sigma": 0.5},
        },
        "rows": rows,
    }


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    rows = results["rows"]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    style.style_axes(ax)
    x = np.arange(len(PERCENTILES))
    width = 0.24
    for j, (row, color) in enumerate(zip(rows, SERIES_COLORS)):
        vals = [row[f"p{p}"] for p in PERCENTILES]
        bars = ax.bar(
            x + (j - 1) * (width + 0.03),
            vals,
            width,
            color=color,
            label=f"{row['shadow_rate']:.0%} shadow",
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 2,
                f"{v:.0f}",
                ha="center",
                color=style.INK,
                fontsize=9,
            )
    ax.set_xticks(x, [f"p{p}" for p in PERCENTILES])
    ax.set_ylabel("User-facing latency (ms)")
    cfg = results["config"]
    style.titles(
        ax,
        "Shadowing adds no user-facing latency",
        f"Identical seeded arrivals and champion latency draws across runs; only shadow rate "
        f"varies\n{cfg['n_requests']:,} requests at {cfg['rate_rps']:.0f} rps · champion "
        f"80ms median, challenger deliberately slower at 120ms median",
    )
    ax.legend(frameon=False, loc="upper left", labelcolor=style.INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "shadow_overhead.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = asyncio.run(run(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "shadow_overhead.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
