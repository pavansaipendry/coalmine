"""Phase 6 experiment: load-test the HTTP front door.

Starts the real API server (uvicorn, in-process) and drives POST /serve with
a concurrent httpx client — the user path routes, logs events, and feeds the
async judging fleet on every request. Reports sustained RPS and user-facing
latency percentiles. deploy/k6-load.js is the equivalent k6 script for
external runs.

Run:  python -m coalmine.experiments.http_load
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import uvicorn

from coalmine.api.server import ServingApp, create_app
from coalmine.experiments import style

PORT = 8143
WARMUP = 300
LIGHT_REQUESTS = 1_200
LIGHT_CONCURRENCY = 4
REQUESTS = 4_000
CONCURRENCY = 32


async def run(seed: int) -> dict:
    app = create_app(ServingApp(seed=seed, n_judges=2))
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    latencies_ms: list[float] = []
    limits = httpx.Limits(
        max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY
    )
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{PORT}", limits=limits
    ) as client:

        async def one() -> float:
            t0 = time.perf_counter()
            response = await client.post("/serve", json={})
            response.raise_for_status()
            return (time.perf_counter() - t0) * 1000.0

        async def batch(n: int, concurrency: int) -> list[float]:
            out: list[float] = []
            pending: set[asyncio.Task] = set()
            issued = 0
            while issued < n or pending:
                while issued < n and len(pending) < concurrency:
                    pending.add(asyncio.create_task(one()))
                    issued += 1
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                out.extend(t.result() for t in done)
            return out

        await batch(WARMUP, CONCURRENCY)
        # Two operating points: service latency under light load, then
        # throughput at saturation (where latency is queueing, by design).
        latencies_ms = await batch(LIGHT_REQUESTS, LIGHT_CONCURRENCY)
        started = time.perf_counter()
        saturated = await batch(REQUESTS, CONCURRENCY)
        wall = time.perf_counter() - started
        state = (await client.get("/state")).json()

    server.should_exit = True
    await server_task

    arr = np.array(latencies_ms)
    sat = np.array(saturated)
    results = {
        "config": {
            "light_requests": LIGHT_REQUESTS,
            "light_concurrency": LIGHT_CONCURRENCY,
            "requests": REQUESTS,
            "concurrency": CONCURRENCY,
            "warmup": WARMUP,
            "seed": seed,
        },
        "rps": REQUESTS / wall,
        "wall_seconds": wall,
        "latency_ms": {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        },
        "saturated_latency_ms": {
            "p50": float(np.percentile(sat, 50)),
            "p99": float(np.percentile(sat, 99)),
        },
        "verdicts_during_test": state["verdicts_seen"],
    }
    print(
        f"  light load (c={LIGHT_CONCURRENCY}): p50 {results['latency_ms']['p50']:.1f}ms, "
        f"p99 {results['latency_ms']['p99']:.1f}ms · saturated (c={CONCURRENCY}): "
        f"{results['rps']:,.0f} req/s · {state['verdicts_seen']} verdicts judged behind it"
    )
    return results


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    style.style_axes(ax)
    labels = ["p50", "p95", "p99"]
    values = [results["latency_ms"][k] for k in labels]
    bars = ax.bar(labels, values, width=0.45, color=style.SERIES_1)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v * 1.02,
            f"{v:.1f}",
            ha="center",
            color=style.INK,
            fontsize=10.5,
            fontweight="bold",
        )
    ax.set_ylabel("User-facing latency over HTTP (ms)")
    style.titles(
        ax,
        "The front door under load",
        f"Light load (c={results['config']['light_concurrency']}) latency · saturated: "
        f"{results['rps']:,.0f} req/s at c={results['config']['concurrency']}, one process\n"
        f"every request logged + shadow-sampled + judged async — "
        f"{results['verdicts_during_test']:,} verdicts during the test",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "http_load.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = asyncio.run(run(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "http_load.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
