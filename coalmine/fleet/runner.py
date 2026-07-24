"""FleetRunner: wire the services to a bus and a store, run a scenario to
completion, and (for the chaos experiment) kill and restart judge workers
mid-flight.

Completion is deterministic: the shadow-sampled pair count is a pure function
of (seed, shadow_rate), so the runner waits until exactly that many verdicts
are processed, drains, flushes every sink, and stops. Because all effects are
idempotent and content-deterministic, a run's final state is comparable
across bus implementations and across chaos schedules by fingerprint.
"""

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from coalmine.core.events import VERDICT_RECORDED
from coalmine.core.store import EventStore
from coalmine.fleet import metrics
from coalmine.fleet.bus import Bus
from coalmine.fleet.drift import InputDriftMonitor
from coalmine.fleet.services import (
    CanaryController,
    DecisionService,
    DriftService,
    JudgeWorker,
    RouterService,
    TrafficService,
    _Sink,
)
from coalmine.judging.judges import PairwiseJudge
from coalmine.serving.backends import (
    MixtureBackend,
    PoolBackend,
    request_rng,
    step_schedule,
    topic_step_schedule,
)
from coalmine.stats.cusum import CUSUM
from coalmine.stats.msprt import MixtureSPRT
from coalmine.traffic.dataset import Query
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario


def expected_pairs(scenario: Scenario) -> int:
    return sum(
        request_rng(scenario.seed, i, "__router__").random() < scenario.shadow_rate
        for i in range(scenario.n_requests)
    )


@dataclass
class FleetResult:
    run_id: str
    wall_seconds: float
    requests_per_second: float
    verdicts: int
    quality_alarm_at: int | None
    promoted_at: int | None
    drift_gated_at: int | None
    worker_verdicts: dict[str, int]
    pair_latency_ms: dict[str, float] = field(default_factory=dict)


class FleetRunner:
    def __init__(
        self,
        bus: Bus,
        store: EventStore,
        scenario: Scenario,
        queries: list[Query],
        pool: ResponsePool,
        judge: PairwiseJudge,
        cusum_h: float,
        n_judges: int = 3,
        drift_monitor: InputDriftMonitor | None = None,
        cusum_p1: float = 0.45,
        reclaim_idle_ms: int = 800,
        lifecycle=None,  # CanaryLifecycle -> run the canary controller instead
    ):
        self.bus = bus
        self.store = store
        self.scenario = scenario
        self.judge = judge
        self.n_judges = n_judges
        self.reclaim_idle_ms = reclaim_idle_ms

        run_id, seed = scenario.run_id, scenario.seed
        if scenario.regressed_topic is not None:
            schedule = topic_step_schedule(
                scenario.epsilon, scenario.changepoint or 0, scenario.regressed_topic
            )
        else:
            schedule = step_schedule(scenario.epsilon, scenario.changepoint)
        champion = PoolBackend("champion", pool, "good", seed)
        challenger = MixtureBackend(
            "challenger",
            PoolBackend("challenger", pool, "good", seed),
            PoolBackend("challenger", pool, "bad", seed),
            schedule,
            seed,
        )
        store_lock = asyncio.Lock()
        self.traffic = TrafficService(
            bus, queries, scenario, _Sink(store, run_id, "traffic", store_lock)
        )
        self.router = RouterService(
            bus,
            _Sink(store, run_id, "router", store_lock),
            champion,
            challenger,
            scenario.shadow_rate,
            seed,
            {q.id: q for q in queries},
            consume_control=lifecycle is not None,
        )
        self.judge_sink = _Sink(store, run_id, "judge", store_lock)
        if lifecycle is not None:
            self.decision = CanaryController(
                bus, _Sink(store, run_id, "canary", store_lock), lifecycle
            )
        else:
            self.decision = DecisionService(
                bus,
                _Sink(store, run_id, "decision", store_lock),
                CUSUM(p0=0.5, p1=cusum_p1, h=cusum_h),
                MixtureSPRT(p0=0.5, alpha=0.05),
            )
        self.drift = (
            DriftService(bus, _Sink(store, run_id, "drift", store_lock), drift_monitor)
            if drift_monitor
            else None
        )
        self.workers: dict[str, JudgeWorker] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def _spawn_worker(self, name: str) -> None:
        worker = JudgeWorker(
            self.bus,
            self.judge_sink,
            self.judge,
            self.scenario.seed,
            name,
            reclaim_idle_ms=self.reclaim_idle_ms,
        )
        self.workers[name] = worker
        self._tasks[name] = asyncio.create_task(worker.run())

    async def kill_worker(self, name: str) -> None:
        """Chaos: cancel a worker wherever it is — consumed-but-unacked
        messages go pending and get reclaimed by the survivors."""
        task = self._tasks.pop(name)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        metrics.WORKER_RESTARTS.inc()

    def restart_worker(self, name: str) -> None:
        self._spawn_worker(name)

    async def run(self, chaos=None, timeout: float = 300.0) -> FleetResult:
        """Run the scenario to completion. `chaos` is an optional async
        callable receiving this runner (kill/restart workers while running)."""
        started = time.perf_counter()
        target = expected_pairs(self.scenario)
        self._tasks["traffic"] = asyncio.create_task(self.traffic.run())
        self._tasks["router"] = asyncio.create_task(self.router.run())
        self._tasks["decision"] = asyncio.create_task(self.decision.run())
        if self.drift:
            self._tasks["drift"] = asyncio.create_task(self.drift.run())
        for j in range(self.n_judges):
            self._spawn_worker(f"judge-{j}")
        chaos_task = asyncio.create_task(chaos(self)) if chaos else None

        deadline = started + timeout
        while self.decision.processed_pairs < target:
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"fleet stalled: {self.decision.processed_pairs}/{target} verdicts"
                )
            await asyncio.sleep(0.05)
        if chaos_task:
            await chaos_task

        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        for sink in (
            self.router.sink,
            self.judge_sink,
            self.decision.sink,
            *((self.drift.sink,) if self.drift else ()),
        ):
            await sink.flush()

        wall = time.perf_counter() - started
        verdict_count = sum(
            1 for e in self.store.read(self.scenario.run_id) if e.type == VERDICT_RECORDED
        )
        latencies = np.array(self.decision.pair_latencies) * 1000.0
        return FleetResult(
            run_id=self.scenario.run_id,
            wall_seconds=wall,
            requests_per_second=self.scenario.n_requests / wall,
            verdicts=verdict_count,
            quality_alarm_at=getattr(self.decision, "quality_alarm_at", None),
            promoted_at=getattr(self.decision, "promoted_at", None),
            drift_gated_at=getattr(self.decision, "drift_gated_at", None),
            worker_verdicts={n: w.judged for n, w in self.workers.items()},
            pair_latency_ms=(
                {
                    "p50": float(np.percentile(latencies, 50)),
                    "p99": float(np.percentile(latencies, 99)),
                }
                if len(latencies)
                else {}
            ),
        )
