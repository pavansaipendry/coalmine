"""Fleet integration on the in-memory bus: the full service pipeline runs a
scenario to completion, is content-deterministic, and — the chaos property —
converges to an identical final state when judge workers are killed mid-run."""

import asyncio

import numpy as np
import pytest

from coalmine.core.events import VERDICT_RECORDED
from coalmine.core.store import SQLiteEventStore
from coalmine.fleet.bus import InMemoryBus
from coalmine.fleet.runner import FleetRunner, expected_pairs
from coalmine.judging.judges import NoisyOracleJudge
from coalmine.sim.runner import calibrate_cusum_threshold
from coalmine.sim.verdicts import gen_wins
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario


@pytest.fixture(scope="module")
def cusum_h():
    rng = np.random.default_rng(3)
    return calibrate_cusum_threshold(gen_wins(rng, 2000, 2000, 0.5), 0.5, 0.45, 0.05)


def _scenario(run_id: str, n: int = 1200) -> Scenario:
    return Scenario(
        run_id=run_id, seed=17, n_requests=n, shadow_rate=0.5, epsilon=0.4, changepoint=n // 3
    )


async def _run_fleet(tmp_path, run_id, chaos=None, db_name=None, n_judges=3, cusum_h=4.0):
    queries = synthetic_queries(60)
    pool = ResponsePool.build_synthetic(queries)
    store = SQLiteEventStore(tmp_path / (db_name or f"{run_id}.db"))
    runner = FleetRunner(
        bus=InMemoryBus(),
        store=store,
        scenario=_scenario(run_id),
        queries=queries,
        pool=pool,
        judge=NoisyOracleJudge(accuracy=0.85, tie_rate=0.1),
        cusum_h=cusum_h,
        n_judges=n_judges,
        reclaim_idle_ms=200,
    )
    result = await runner.run(chaos=chaos, timeout=120.0)
    return store, result


def _verdict_fingerprint(store, run_id):
    events = store.read(run_id)
    return sorted(
        (e.payload["request_index"], e.payload["winner"], e.payload["first_shown"])
        for e in events
        if e.type == VERDICT_RECORDED
    )


async def test_fleet_completes_and_detects(tmp_path, cusum_h):
    store, result = await _run_fleet(tmp_path, "fleet-basic", cusum_h=cusum_h)
    target = expected_pairs(_scenario("fleet-basic"))
    assert result.verdicts == target
    assert result.quality_alarm_at is not None, "40% regression must alarm"
    assert result.quality_alarm_at > 400  # after the changepoint
    assert result.promoted_at is None  # a worse challenger is never promoted
    assert sum(result.worker_verdicts.values()) == target
    # Work is shared across the group; requiring every worker to win the race
    # for at least one message would be scheduler-dependent, so assert >= 2.
    assert sum(v > 0 for v in result.worker_verdicts.values()) >= 2
    store.close()


async def test_fleet_is_content_deterministic(tmp_path, cusum_h):
    store_a, result_a = await _run_fleet(tmp_path, "fleet-a", cusum_h=cusum_h)
    store_b, result_b = await _run_fleet(tmp_path, "fleet-a", db_name="b.db", cusum_h=cusum_h)
    assert _verdict_fingerprint(store_a, "fleet-a") == _verdict_fingerprint(store_b, "fleet-a")
    assert result_a.quality_alarm_at == result_b.quality_alarm_at
    store_a.close()
    store_b.close()


async def test_chaos_kill_and_restart_converges(tmp_path, cusum_h):
    async def chaos(runner):
        await asyncio.sleep(0.3)
        await runner.kill_worker("judge-0")
        await asyncio.sleep(0.3)
        runner.restart_worker("judge-0")
        await asyncio.sleep(0.2)
        await runner.kill_worker("judge-1")

    baseline_store, baseline = await _run_fleet(tmp_path, "fleet-c", cusum_h=cusum_h)
    chaos_store, chaotic = await _run_fleet(
        tmp_path, "fleet-c", chaos=chaos, db_name="chaos.db", cusum_h=cusum_h
    )
    # Kills, redelivery, and duplicate judging must not change the outcome:
    # same verdicts (exactly once, by idempotent seq), same alarm.
    assert _verdict_fingerprint(chaos_store, "fleet-c") == _verdict_fingerprint(
        baseline_store, "fleet-c"
    )
    assert chaotic.verdicts == baseline.verdicts
    assert chaotic.quality_alarm_at == baseline.quality_alarm_at
    baseline_store.close()
    chaos_store.close()
