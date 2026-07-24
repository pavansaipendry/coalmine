"""Phase 5 headline: the fleet on real Redis, with chaos.

The full multi-service pipeline — traffic, router, three judge workers in one
consumer group, drift monitor, decision engine — runs a 20,000-request
regression scenario over Redis Streams, twice: a clean baseline, then a chaos
run in which every judge worker is killed mid-flight (and two restarted).
Consumed-but-unacked messages are reclaimed by the survivors; because every
effect is idempotent (seq derived from request identity) and deterministic
(per-request RNG streams), the chaos run must converge to byte-identical
verdicts and the same alarm — verified by fingerprint, not asserted.

Falls back to the in-memory bus if no Redis is reachable (same semantics,
same convergence claim; the Redis path also runs in CI).

Run:  python -m coalmine.experiments.fleet_run
"""

import argparse
import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.core.events import VERDICT_RECORDED
from coalmine.core.store import SQLiteEventStore
from coalmine.experiments import style
from coalmine.fleet.bus import InMemoryBus, RedisStreamBus
from coalmine.fleet.drift import InputDriftMonitor, calibrate_psi_threshold
from coalmine.fleet.runner import FleetRunner
from coalmine.judging.judges import NoisyOracleJudge
from coalmine.sim.runner import calibrate_cusum_threshold
from coalmine.sim.verdicts import gen_wins
from coalmine.traffic.dataset import DEFAULT_TOPICS, synthetic_queries
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario

N_REQUESTS = 20_000
CHANGEPOINT = 8_000
EPSILON = 0.15
SHADOW_RATE = 0.25
N_JUDGES = 3
JUDGE = dict(accuracy=0.85, tie_rate=0.10, position_bias=0.15)


async def _make_bus():
    url = os.environ.get("COALMINE_REDIS_URL", "redis://localhost:6379/0")
    bus = RedisStreamBus(url, prefix=f"coalmine:fleet:{uuid.uuid4().hex[:12]}")
    try:
        await bus.redis.ping()
        return bus, "redis"
    except Exception:
        await bus.close()
        return InMemoryBus(), "memory"


def _fingerprint(store, run_id):
    return sorted(
        (e.payload["request_index"], e.payload["winner"], e.payload["first_shown"])
        for e in store.read(run_id)
        if e.type == VERDICT_RECORDED
    )


async def _one_run(scenario, queries, pool, cusum_h, drift_threshold, chaos, db_path):
    bus, bus_kind = await _make_bus()
    store = SQLiteEventStore(db_path)
    monitor = InputDriftMonitor(
        sorted(DEFAULT_TOPICS), window=1_000, stride=100, threshold=drift_threshold
    )
    runner = FleetRunner(
        bus=bus,
        store=store,
        scenario=scenario,
        queries=queries,
        pool=pool,
        judge=NoisyOracleJudge(**JUDGE),
        cusum_h=cusum_h,
        n_judges=N_JUDGES,
        drift_monitor=monitor,
    )
    try:
        result = await runner.run(chaos=chaos, timeout=600.0)
    finally:
        if isinstance(bus, RedisStreamBus):
            await bus.cleanup()
        await bus.close()
    return store, result, bus_kind


async def run(seed: int) -> dict:
    queries = synthetic_queries(400)
    pool = ResponsePool.build_synthetic(queries)
    scenario = Scenario(
        run_id="fleet-main",
        seed=seed,
        n_requests=N_REQUESTS,
        shadow_rate=SHADOW_RATE,
        epsilon=EPSILON,
        changepoint=CHANGEPOINT,
    )
    rng = np.random.default_rng(seed + 1)
    horizon = int(N_REQUESTS * SHADOW_RATE * 0.92)
    cusum_h = calibrate_cusum_threshold(gen_wins(rng, 4_000, horizon, 0.5), 0.5, 0.45, 0.05)
    drift_threshold = calibrate_psi_threshold(
        np.full(4, 0.25), 1_000, 100, N_REQUESTS - 1_000, 0.05, trials=400, seed=seed
    )

    tmp = Path(tempfile.mkdtemp())
    print("  baseline run ...")
    base_store, base, bus_kind = await _one_run(
        scenario, queries, pool, cusum_h, drift_threshold, None, tmp / "base.db"
    )
    print(
        f"  [{bus_kind}] {base.requests_per_second:,.0f} req/s · verdicts {base.verdicts} · "
        f"quality alarm @ {base.quality_alarm_at} · drift alarms: "
        f"{base.drift_gated_at is not None}"
    )

    async def chaos(runner):
        for j in range(N_JUDGES):
            await asyncio.sleep(2.0)
            await runner.kill_worker(f"judge-{j}")
            if j < N_JUDGES - 1:
                await asyncio.sleep(0.4)
                runner.restart_worker(f"judge-{j}")

    print("  chaos run (kill every judge worker mid-flight) ...")
    chaos_store, chaotic, _ = await _one_run(
        scenario, queries, pool, cusum_h, drift_threshold, chaos, tmp / "chaos.db"
    )
    fp_base, fp_chaos = _fingerprint(base_store, "fleet-main"), _fingerprint(
        chaos_store, "fleet-main"
    )
    converged = fp_base == fp_chaos and base.quality_alarm_at == chaotic.quality_alarm_at
    print(
        f"  chaos: verdicts {chaotic.verdicts} · alarm @ {chaotic.quality_alarm_at} · "
        f"converged to baseline: {converged}"
    )
    base_store.close()
    chaos_store.close()

    return {
        "config": {
            "n_requests": N_REQUESTS,
            "changepoint": CHANGEPOINT,
            "epsilon": EPSILON,
            "shadow_rate": SHADOW_RATE,
            "n_judges": N_JUDGES,
            "judge": JUDGE,
            "bus": bus_kind,
            "seed": seed,
        },
        "baseline": {
            "rps": base.requests_per_second,
            "wall_seconds": base.wall_seconds,
            "verdicts": base.verdicts,
            "quality_alarm_at": base.quality_alarm_at,
            "drift_alarmed": base.drift_gated_at is not None,
            "worker_verdicts": base.worker_verdicts,
            "pair_latency_ms": base.pair_latency_ms,
        },
        "chaos": {
            "rps": chaotic.requests_per_second,
            "verdicts": chaotic.verdicts,
            "quality_alarm_at": chaotic.quality_alarm_at,
            "worker_verdicts": chaotic.worker_verdicts,
            "pair_latency_ms": chaotic.pair_latency_ms,
        },
        "converged": converged,
    }


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.5))
    for ax in (ax1, ax2):
        style.style_axes(ax)

    labels = ["p50", "p99"]
    base_lat = [results["baseline"]["pair_latency_ms"].get(k, 0) for k in labels]
    chaos_lat = [results["chaos"]["pair_latency_ms"].get(k, 0) for k in labels]
    x = np.arange(2)
    width = 0.32
    for offset, vals, color, name in (
        (-width / 2, base_lat, style.SERIES_1, "Baseline"),
        (width / 2, chaos_lat, style.SERIES_2, "Chaos run"),
    ):
        bars = ax1.bar(x + offset, vals, width * 0.94, color=color, label=name)
        for bar, v in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, v * 1.02, f"{v:.2f}",
                ha="center", color=style.INK, fontsize=9.5,
            )
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Pair → verdict latency (ms)")
    cfg = results["config"]
    style.titles(
        ax1,
        "Queue latency under chaos",
        f"{cfg['bus']} bus · {results['baseline']['rps']:,.0f} req/s sustained · "
        f"{cfg['n_judges']} judge workers",
    )
    ax1.legend(frameon=False, loc="upper left", labelcolor=style.INK_2, fontsize=9)

    workers = sorted(results["baseline"]["worker_verdicts"])
    x = np.arange(len(workers))
    for offset, key, color, name in (
        (-width / 2, "baseline", style.SERIES_1, "Baseline"),
        (width / 2, "chaos", style.SERIES_2, "Chaos run"),
    ):
        vals = [results[key]["worker_verdicts"].get(w, 0) for w in workers]
        bars = ax2.bar(x + offset, vals, width * 0.94, color=color, label=name)
        for bar, v in zip(bars, vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, v + 20, f"{v}",
                ha="center", color=style.INK, fontsize=9.5,
            )
    ax2.set_xticks(x, workers)
    ax2.set_ylabel("Verdicts judged")
    style.titles(
        ax2,
        "Every worker killed; the group absorbs it",
        "Reclaim + idempotent appends: byte-identical verdicts,\n"
        f"same alarm @ {results['chaos']['quality_alarm_at']:,} — "
        f"converged: {results['converged']}",
    )
    fig.tight_layout(w_pad=3.0)
    fig.savefig(out_dir / "fleet_run.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = asyncio.run(run(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fleet_run.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
