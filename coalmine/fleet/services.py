"""The fleet: traffic, router, judge workers, drift monitor, decision engine.

Each service is an independent consumer on the bus; judge workers share one
consumer group, so pairs load-balance across them and a killed worker's
unacked messages are reclaimed by the survivors.

Exactly-once effects without coordination: every event's seq is derived from
request identity (seq bands below), and every service's output is a pure
function of its input message (the per-request RNG discipline from Phase 2).
A redelivered message therefore re-produces byte-identical events that the
idempotent store drops — so any interleaving of crashes and redeliveries
converges to the same final state, which the chaos experiment verifies by
fingerprint.

Seq bands (per request index i):
    10*i + 0  request_received      10*i + 5  verdict_recorded
    10*i + 1  champion_response     10*i + 6  quality alarm / promotion
    10*i + 2  shadow_response       10*i + 7  drift alarm / test reset
Run-level events live at 10*n_requests + k.
"""

import asyncio
import time

import numpy as np

from coalmine.core.events import (
    CHAMPION_RESPONSE,
    REQUEST_RECEIVED,
    SHADOW_RESPONSE,
    VERDICT_RECORDED,
    Event,
)
from coalmine.core.store import EventStore
from coalmine.fleet import metrics
from coalmine.fleet.bus import Bus
from coalmine.fleet.drift import InputDriftMonitor
from coalmine.judging.judges import PairwiseJudge, compare_randomized
from coalmine.serving.backends import Request, ResponseBackend, request_rng
from coalmine.stats.cusum import CUSUM
from coalmine.stats.msprt import MixtureSPRT
from coalmine.traffic.dataset import Query
from coalmine.traffic.simulator import _weights_vector

REQUESTS, PAIRS, VERDICTS, ALARMS, CONTROL = (
    "requests",
    "pairs",
    "verdicts",
    "alarms",
    "control",
)

DRIFT_ALARM = "drift_alarm"
QUALITY_ALARM = "quality_alarm"
PROMOTION = "promotion"
TEST_RESET = "test_reset"
CANARY_TRANSITION = "canary_transition"
SHARE_UPDATE = "share_update"


class _Sink:
    """Buffered writer around the idempotent store. Services flush
    concurrently but one store connection is not thread-safe under
    simultaneous use — all sinks of a fleet share one lock that serializes
    the actual writes."""

    def __init__(
        self,
        store: EventStore,
        run_id: str,
        service: str,
        lock: asyncio.Lock | None = None,
        batch: int = 128,
    ):
        self.store = store
        self.run_id = run_id
        self.service = service
        self.lock = lock or asyncio.Lock()
        self.batch = batch
        self._buffer: list[Event] = []

    async def emit(self, seq: int, type_: str, payload: dict) -> None:
        self._buffer.append(Event(self.run_id, seq, type_, payload, time.time()))
        metrics.EVENTS_WRITTEN.labels(self.service).inc()
        if len(self._buffer) >= self.batch:
            await self.flush()

    async def flush(self) -> None:
        if self._buffer:
            batch = self._buffer
            self._buffer = []
            async with self.lock:
                await asyncio.to_thread(self.store.append_many, batch)


class TrafficService:
    """Publishes the scenario's request stream to the bus (sequential mode)."""

    def __init__(self, bus: Bus, queries: list[Query], scenario, sink: _Sink):
        self.bus = bus
        self.queries = queries
        self.scenario = scenario
        self.sink = sink
        by_topic: dict[str, list[Query]] = {}
        for q in queries:
            by_topic.setdefault(q.topic, []).append(q)
        self.topics = sorted(by_topic)
        self.by_topic = by_topic

    def _pick(self, index: int) -> Query:
        s = self.scenario
        rng = request_rng(s.seed, index, "__traffic__")
        shifted = s.topic_shift_at is not None and index >= s.topic_shift_at
        weights = _weights_vector(
            self.topics, (s.topic_weights_after or s.topic_weights) if shifted else s.topic_weights
        )
        topic = self.topics[int(rng.choice(len(self.topics), p=np.asarray(weights)))]
        pool = self.by_topic[topic]
        return pool[int(rng.integers(len(pool)))]

    async def run(self) -> None:
        for i in range(self.scenario.n_requests):
            query = self._pick(i)
            await self.bus.publish(
                REQUESTS,
                {"index": i, "query_id": query.id, "text": query.text, "topic": query.topic},
            )
            metrics.MESSAGES_PROCESSED.labels("traffic").inc()
        await self.sink.flush()


class RouterService:
    """Consumes requests; serves champion, shadow-samples the challenger, and
    publishes judged-pair candidates."""

    def __init__(
        self,
        bus: Bus,
        store_sink: _Sink,
        champion: ResponseBackend,
        challenger: ResponseBackend,
        shadow_rate: float,
        seed: int,
        queries_by_id: dict[str, Query],
        consume_control: bool = False,
    ):
        self.bus = bus
        self.sink = store_sink
        self.champion = champion
        self.challenger = challenger
        self.shadow_rate = shadow_rate
        self.seed = seed
        self.queries_by_id = queries_by_id
        self.consume_control = consume_control
        self.user_share = 0.0  # fraction of user traffic the challenger serves

    async def run(self, consumer: str = "router-0") -> None:
        streams = [REQUESTS, CONTROL] if self.consume_control else [REQUESTS]
        async for message in self.bus.consume(streams, "router", consumer):
            p = message.payload
            if message.stream == CONTROL:
                if p.get("kind") == SHARE_UPDATE:
                    self.user_share = p["share"]
                await self.bus.ack(message, "router")
                continue
            t0 = time.perf_counter()
            index, query = p["index"], self.queries_by_id[p["query_id"]]
            request = Request(index, query)
            canary_serves = (
                request_rng(self.seed, index, "__canary__").random() < self.user_share
            )
            await self.sink.emit(
                10 * index + 0,
                REQUEST_RECEIVED,
                {
                    "request_index": index,
                    "query_id": query.id,
                    "query_text": query.text,
                    "topic": query.topic,
                    "served_by": "challenger" if canary_serves else "champion",
                },
            )
            champ = await self.champion.generate(request)
            await self.sink.emit(
                10 * index + 1,
                CHAMPION_RESPONSE,
                {
                    "request_index": index,
                    "query_id": query.id,
                    "config_id": champ.config_id,
                    "source_pool": champ.source_pool,
                    "latency_ms": champ.latency_ms,
                    "text": champ.text,
                },
            )
            sampled = request_rng(self.seed, index, "__router__").random() < self.shadow_rate
            if sampled:
                shadow = await self.challenger.generate(request)
                await self.sink.emit(
                    10 * index + 2,
                    SHADOW_RESPONSE,
                    {
                        "request_index": index,
                        "query_id": query.id,
                        "config_id": shadow.config_id,
                        "source_pool": shadow.source_pool,
                        "latency_ms": shadow.latency_ms,
                        "text": shadow.text,
                        **shadow.meta,
                    },
                )
                await self.bus.publish(
                    PAIRS,
                    {
                        "index": index,
                        "query_text": query.text,
                        "topic": p["topic"],
                        "a_text": champ.text,
                        "a_pool": champ.source_pool,
                        "b_text": shadow.text,
                        "b_pool": shadow.source_pool,
                        "emitted_at": time.time(),
                    },
                )
            await self.bus.ack(message, "router")
            metrics.MESSAGES_PROCESSED.labels("router").inc()
            metrics.STAGE_LATENCY.labels("router").observe(time.perf_counter() - t0)


class JudgeWorker:
    """One member of the judge consumer group. Killing a worker mid-message
    leaves it pending; a peer reclaims and re-judges it deterministically."""

    def __init__(
        self,
        bus: Bus,
        sink: _Sink,
        judge: PairwiseJudge,
        seed: int,
        name: str,
        reclaim_idle_ms: int = 800,
    ):
        self.bus = bus
        self.sink = sink
        self.judge = judge
        self.seed = seed
        self.name = name
        self.reclaim_idle_ms = reclaim_idle_ms
        self.judged = 0

    async def run(self) -> None:
        async for message in self.bus.consume(
            [PAIRS], "judges", self.name, reclaim_idle_ms=self.reclaim_idle_ms
        ):
            t0 = time.perf_counter()
            p = message.payload
            index = p["index"]
            rng = request_rng(self.seed, index, f"__judge__{self.judge.judge_id}")
            result = await compare_randomized(
                self.judge,
                p["query_text"],
                {"text": p["a_text"], "source_pool": p["a_pool"]},
                {"text": p["b_text"], "source_pool": p["b_pool"]},
                rng,
            )
            payload = {
                "request_index": index,
                "config_a": "champion",
                "config_b": "challenger",
                "winner": result.winner,
                "first_shown": result.first_shown,
                "judge_id": self.judge.judge_id,
                "topic": p["topic"],
                "worker": self.name,
            }
            await self.sink.emit(10 * index + 5, VERDICT_RECORDED, payload)
            await self.bus.publish(
                VERDICTS,
                {**payload, "pair_emitted_at": p.get("emitted_at"), "judged_at": time.time()},
            )
            await self.bus.ack(message, "judges")
            self.judged += 1
            metrics.MESSAGES_PROCESSED.labels("judge").inc()
            metrics.STAGE_LATENCY.labels("judge").observe(time.perf_counter() - t0)


class DriftService:
    """Watches the request stream's topic mix; a PSI alarm is published to the
    alarms stream, where the decision service treats it as a gate."""

    def __init__(self, bus: Bus, sink: _Sink, monitor: InputDriftMonitor):
        self.bus = bus
        self.sink = sink
        self.monitor = monitor

    async def run(self, consumer: str = "drift-0") -> None:
        async for message in self.bus.consume([REQUESTS], "drift", consumer):
            p = message.payload
            already = self.monitor.alarmed_at is not None
            alarmed = self.monitor.observe(p["topic"])
            if alarmed and not already:
                payload = {
                    "request_index": p["index"],
                    "kind": DRIFT_ALARM,
                    "psi": self.monitor.last_psi,
                    "threshold": self.monitor.threshold,
                }
                await self.sink.emit(10 * p["index"] + 7, DRIFT_ALARM, payload)
                await self.bus.publish(ALARMS, payload)
                metrics.ALARMS.labels(DRIFT_ALARM).inc()
            await self.bus.ack(message, "drift")
            metrics.MESSAGES_PROCESSED.labels("drift").inc()


class CanaryController:
    """The capstone service: consumes verdicts, drives the canary lifecycle
    (per-stage non-inferiority gates + continuous rollback CUSUM), publishes
    share updates on the control stream for the router, and writes every
    transition to the audit log."""

    def __init__(self, bus: Bus, sink: _Sink, lifecycle):
        self.bus = bus
        self.sink = sink
        self.lifecycle = lifecycle
        self.processed_pairs = 0
        self.pair_latencies: list[float] = []

    async def run(self, consumer: str = "canary-0") -> None:
        await self.bus.publish(
            CONTROL, {"kind": SHARE_UPDATE, "share": self.lifecycle.share}
        )
        async for message in self.bus.consume([VERDICTS], "decision", consumer):
            p = message.payload
            self.processed_pairs += 1
            if p.get("pair_emitted_at") and p.get("judged_at"):
                self.pair_latencies.append(p["judged_at"] - p["pair_emitted_at"])
            if p["winner"] != "tie":
                index = p["request_index"]
                transition = self.lifecycle.update(p["winner"] == "b", index)
                if transition is not None:
                    payload = {
                        "request_index": index,
                        "kind": transition.kind,
                        "stage": transition.stage,
                        "share": transition.share,
                        "verdicts_in_stage": transition.verdicts_in_stage,
                    }
                    await self.sink.emit(10 * index + 9, CANARY_TRANSITION, payload)
                    await self.bus.publish(
                        CONTROL, {"kind": SHARE_UPDATE, "share": transition.share}
                    )
                    metrics.ALARMS.labels(transition.kind).inc()
            await self.bus.ack(message, "decision")
            metrics.MESSAGES_PROCESSED.labels("canary").inc()


class DecisionService:
    """Consumes verdicts and drift alarms. CUSUM raises the quality alarm;
    the mixture SPRT gates promotion; a drift alarm resets both in-flight
    tests (their iid assumption just died) and re-baselines."""

    def __init__(
        self,
        bus: Bus,
        sink: _Sink,
        cusum: CUSUM,
        promotion: MixtureSPRT,
    ):
        self.bus = bus
        self.sink = sink
        self.cusum = cusum
        self.promotion = promotion
        self.quality_alarm_at: int | None = None
        self.promoted_at: int | None = None
        self.drift_gated_at: int | None = None
        self.verdicts_seen = 0
        self.processed_pairs = 0  # every verdict message, ties included
        self.pair_latencies: list[float] = []  # pair published -> verdict published

    async def run(self, consumer: str = "decision-0") -> None:
        async for message in self.bus.consume([VERDICTS, ALARMS], "decision", consumer):
            p = message.payload
            if message.stream == ALARMS and p.get("kind") == DRIFT_ALARM:
                self.cusum.reset()
                self.promotion.reset()
                self.drift_gated_at = p["request_index"]
                await self.sink.emit(
                    10 * p["request_index"] + 8,
                    TEST_RESET,
                    {"request_index": p["request_index"], "cause": DRIFT_ALARM},
                )
            elif message.stream == VERDICTS:
                self.processed_pairs += 1
                if p.get("pair_emitted_at") and p.get("judged_at"):
                    self.pair_latencies.append(p["judged_at"] - p["pair_emitted_at"])
                if p["winner"] == "tie":
                    await self.bus.ack(message, "decision")
                    metrics.MESSAGES_PROCESSED.labels("decision").inc()
                    continue
                t0 = time.perf_counter()
                self.verdicts_seen += 1
                win = p["winner"] == "b"
                index = p["request_index"]
                if self.cusum.update(win) and self.quality_alarm_at is None:
                    self.quality_alarm_at = index
                    payload = {"request_index": index, "kind": QUALITY_ALARM}
                    await self.sink.emit(10 * index + 6, QUALITY_ALARM, payload)
                    await self.bus.publish(ALARMS, payload)
                    metrics.ALARMS.labels(QUALITY_ALARM).inc()
                if self.promotion.update(win) and self.promoted_at is None:
                    self.promoted_at = index
                    await self.sink.emit(
                        10 * index + 6, PROMOTION, {"request_index": index, "kind": PROMOTION}
                    )
                    metrics.ALARMS.labels(PROMOTION).inc()
                metrics.STAGE_LATENCY.labels("decision").observe(time.perf_counter() - t0)
            await self.bus.ack(message, "decision")
            metrics.MESSAGES_PROCESSED.labels("decision").inc()
