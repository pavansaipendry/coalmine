"""Traffic simulator: replays queries through the shadow router as a stream.

Two modes: paced (Poisson arrivals at rate_rps, requests run concurrently
like a real server) and sequential (as fast as possible, for large logged
runs). Scenario knobs: inject an ε-mixture regression into the challenger at
a changepoint, and shift the topic distribution at a changepoint. Everything
lands in the event log; ground truth is recoverable from it afterward.
"""

import asyncio
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from coalmine.core.events import RUN_FINISHED, RUN_STARTED
from coalmine.core.store import EventLog, EventStore
from coalmine.serving.backends import (
    LatencyModel,
    MixtureBackend,
    PoolBackend,
    Request,
    request_rng,
    step_schedule,
    topic_step_schedule,
)
from coalmine.serving.router import ShadowRouter
from coalmine.traffic.dataset import Query
from coalmine.traffic.pools import ResponsePool


@dataclass(frozen=True)
class Scenario:
    run_id: str
    seed: int
    n_requests: int
    rate_rps: float | None = None  # None = sequential, as fast as possible
    shadow_rate: float = 0.2
    epsilon: float = 0.0
    changepoint: int | None = None
    regressed_topic: str | None = None  # None = regression applies to all topics
    topic_weights: dict[str, float] | None = None  # None = uniform
    topic_weights_after: dict[str, float] | None = None
    topic_shift_at: int | None = None
    champion_latency: LatencyModel | None = None
    challenger_latency: LatencyModel | None = None


@dataclass
class SimResult:
    run_id: str
    n_requests: int
    counts: dict[str, int]
    user_latency_ms: list[float] = field(repr=False, default_factory=list)
    wall_seconds: float = 0.0


def _weights_vector(topics: list[str], weights: dict[str, float] | None) -> np.ndarray:
    if weights is None:
        return np.full(len(topics), 1.0 / len(topics))
    v = np.array([weights.get(t, 0.0) for t in topics], dtype=float)
    if v.sum() <= 0:
        raise ValueError("topic weights must have positive mass on known topics")
    return v / v.sum()


def _pick_query(
    scenario: Scenario,
    index: int,
    topics: list[str],
    by_topic: dict[str, list[Query]],
    w_before: np.ndarray,
    w_after: np.ndarray,
) -> Query:
    rng = request_rng(scenario.seed, index, "__traffic__")
    shifted = scenario.topic_shift_at is not None and index >= scenario.topic_shift_at
    weights = w_after if shifted else w_before
    topic = topics[int(rng.choice(len(topics), p=weights))]
    pool = by_topic[topic]
    return pool[int(rng.integers(len(pool)))]


async def run_simulation(
    scenario: Scenario,
    queries: list[Query],
    pool: ResponsePool,
    store: EventStore,
) -> SimResult:
    topics = sorted({q.topic for q in queries})
    by_topic = {t: [q for q in queries if q.topic == t] for t in topics}
    w_before = _weights_vector(topics, scenario.topic_weights)
    w_after = _weights_vector(topics, scenario.topic_weights_after or scenario.topic_weights)

    champion = PoolBackend(
        "champion", pool, "good", scenario.seed, latency=scenario.champion_latency
    )
    if scenario.regressed_topic is not None:
        schedule = topic_step_schedule(
            scenario.epsilon, scenario.changepoint or 0, scenario.regressed_topic
        )
    else:
        schedule = step_schedule(scenario.epsilon, scenario.changepoint)
    challenger = MixtureBackend(
        "challenger",
        PoolBackend("challenger", pool, "good", scenario.seed, scenario.challenger_latency),
        PoolBackend("challenger", pool, "bad", scenario.seed, scenario.challenger_latency),
        schedule,
        scenario.seed,
    )

    log = EventLog(store, scenario.run_id)
    await log.start()
    router = ShadowRouter(champion, [challenger], scenario.shadow_rate, scenario.seed, log)
    latencies: list[float] = []
    started = time.perf_counter()
    log.emit(RUN_STARTED, {"scenario": asdict(scenario), "topics": topics})
    try:
        if scenario.rate_rps is None:
            for i in range(scenario.n_requests):
                request = Request(
                    i, _pick_query(scenario, i, topics, by_topic, w_before, w_after)
                )
                t0 = time.perf_counter()
                await router.route(request)
                latencies.append((time.perf_counter() - t0) * 1000.0)
        else:
            tasks = []

            async def timed_route(request: Request) -> None:
                t0 = time.perf_counter()
                await router.route(request)
                latencies.append((time.perf_counter() - t0) * 1000.0)

            for i in range(scenario.n_requests):
                gap_rng = request_rng(scenario.seed, i, "__arrival__")
                await asyncio.sleep(float(gap_rng.exponential(1.0 / scenario.rate_rps)))
                request = Request(
                    i, _pick_query(scenario, i, topics, by_topic, w_before, w_after)
                )
                tasks.append(asyncio.create_task(timed_route(request)))
            await asyncio.gather(*tasks)
        await router.drain()
    finally:
        wall = time.perf_counter() - started
        log.emit(
            RUN_FINISHED,
            {"n_requests": scenario.n_requests, "counts": dict(log.counts), "wall_seconds": wall},
        )
        await log.close()
    return SimResult(scenario.run_id, scenario.n_requests, dict(log.counts), latencies, wall)
