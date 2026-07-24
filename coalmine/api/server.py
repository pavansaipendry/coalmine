"""The HTTP front door: serve requests, watch the lifecycle, stream the log.

POST /serve is the user path — the handler routes to champion or challenger
per the canary lifecycle's current share and returns the response inline;
the judging pipeline (judge workers + canary controller on the in-process
bus) runs asynchronously behind it, exactly like the fleet. GET /state
summarizes the lifecycle; GET /events/stream tails the event log as SSE; the
dashboard at / renders both live. This is also the k6 / load-test target.

Run:  python -m coalmine.api.server   (then open http://localhost:8123)
"""

import asyncio
import itertools
import json
import tempfile
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from coalmine.canary.lifecycle import CanaryLifecycle
from coalmine.core.events import CHAMPION_RESPONSE, REQUEST_RECEIVED, SHADOW_RESPONSE
from coalmine.core.store import SQLiteEventStore
from coalmine.fleet.bus import InMemoryBus
from coalmine.fleet.services import PAIRS, CanaryController, JudgeWorker, _Sink
from coalmine.judging.judges import NoisyOracleJudge
from coalmine.serving.backends import (
    MixtureBackend,
    PoolBackend,
    Request,
    request_rng,
    step_schedule,
)
from coalmine.sim.runner import calibrate_cusum_threshold
from coalmine.sim.verdicts import gen_wins
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool

RUN_ID = "api-live"


class ServeRequest(BaseModel):
    query_id: str | None = None


class ServingApp:
    """Everything behind the HTTP surface: backends, event log, and the
    in-process judging fleet driving the canary lifecycle."""

    def __init__(
        self,
        seed: int = 2030,
        n_queries: int = 400,
        epsilon: float = 0.0,
        shadow_rate: float = 0.35,
        n_judges: int = 2,
        db_path: str | Path | None = None,
        cusum_h: float | None = None,
    ):
        self.seed = seed
        self.shadow_rate = shadow_rate
        self.queries = synthetic_queries(n_queries)
        pool = ResponsePool.build_synthetic(self.queries)
        self.champion = PoolBackend("champion", pool, "good", seed)
        self.challenger = MixtureBackend(
            "challenger",
            PoolBackend("challenger", pool, "good", seed),
            PoolBackend("challenger", pool, "bad", seed),
            step_schedule(epsilon, 0 if epsilon else None),
            seed,
        )
        if cusum_h is None:
            rng = np.random.default_rng(seed)
            cusum_h = calibrate_cusum_threshold(
                gen_wins(rng, 1_000, 5_000, 0.5), 0.5, 0.45, 0.05
            )
        self.lifecycle = CanaryLifecycle(cusum_h=cusum_h)
        self.store = SQLiteEventStore(
            db_path or Path(tempfile.mkdtemp()) / "api.db"
        )
        self.bus = InMemoryBus()
        lock = asyncio.Lock()
        self.sink = _Sink(self.store, RUN_ID, "api", lock)
        self.judge_sink = _Sink(self.store, RUN_ID, "judge", lock)
        self.controller = CanaryController(
            self.bus, _Sink(self.store, RUN_ID, "canary", lock), self.lifecycle
        )
        self.n_judges = n_judges
        self.recent_wins: deque[bool] = deque(maxlen=500)
        self._index = itertools.count()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        judge = NoisyOracleJudge(accuracy=0.85, tie_rate=0.10, position_bias=0.15)
        for j in range(self.n_judges):
            worker = JudgeWorker(self.bus, self.judge_sink, judge, self.seed, f"judge-{j}")
            self._tasks.append(asyncio.create_task(worker.run()))
        self._tasks.append(asyncio.create_task(self.controller.run()))
        self._tasks.append(asyncio.create_task(self._track_wins()))
        self._tasks.append(asyncio.create_task(self._flusher()))

    async def _flusher(self) -> None:
        # Sinks batch writes for throughput; at API traffic levels the batch
        # may never fill, so flush on a short interval to keep the SSE tail
        # and the log fresh (bounded staleness instead of unbounded).
        while True:
            await asyncio.sleep(0.25)
            for sink in (self.sink, self.judge_sink, self.controller.sink):
                await sink.flush()

    async def _track_wins(self) -> None:
        async for message in self.bus.consume(["verdicts"], "apiwins", "tracker"):
            if message.payload["winner"] != "tie":
                self.recent_wins.append(message.payload["winner"] == "b")
            await self.bus.ack(message, "apiwins")

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for sink in (self.sink, self.judge_sink, self.controller.sink):
            await sink.flush()
        self.store.close()

    async def serve(self, query_id: str | None = None) -> dict:
        index = next(self._index)
        rng = request_rng(self.seed, index, "__api__")
        query = (
            next(q for q in self.queries if q.id == query_id)
            if query_id
            else self.queries[int(rng.integers(len(self.queries)))]
        )
        request = Request(index, query)
        t0 = time.perf_counter()
        canary_serves = (
            request_rng(self.seed, index, "__canary__").random() < self.lifecycle.share
        )
        champ = await self.champion.generate(request)
        shadow = None
        sampled = request_rng(self.seed, index, "__router__").random() < self.shadow_rate
        if sampled or canary_serves:
            shadow = await self.challenger.generate(request)
        user_response = shadow if (canary_serves and shadow) else champ
        latency_ms = (time.perf_counter() - t0) * 1000.0

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
        await self.sink.emit(
            10 * index + 1,
            CHAMPION_RESPONSE,
            {
                "request_index": index,
                "query_id": query.id,
                "config_id": "champion",
                "source_pool": champ.source_pool,
                "latency_ms": champ.latency_ms,
                "text": champ.text,
            },
        )
        if shadow is not None:
            await self.sink.emit(
                10 * index + 2,
                SHADOW_RESPONSE,
                {
                    "request_index": index,
                    "query_id": query.id,
                    "config_id": "challenger",
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
                    "topic": query.topic,
                    "a_text": champ.text,
                    "a_pool": champ.source_pool,
                    "b_text": shadow.text,
                    "b_pool": shadow.source_pool,
                    "emitted_at": time.time(),
                },
            )
        return {
            "request_index": index,
            "query_id": query.id,
            "served_by": "challenger" if canary_serves else "champion",
            "response": user_response.text,
            "latency_ms": round(latency_ms, 3),
        }

    def state(self) -> dict:
        wins = list(self.recent_wins)
        return {
            "stage": self.lifecycle.stage_name,
            "share": self.lifecycle.share,
            "state": self.lifecycle.state,
            "verdicts_seen": self.lifecycle.verdicts_seen,
            "recent_win_rate": (sum(wins) / len(wins)) if wins else None,
            "transitions": [
                {
                    "kind": t.kind,
                    "stage": t.stage,
                    "share": t.share,
                    "request_index": t.request_index,
                }
                for t in self.lifecycle.transitions
            ],
        }


def create_app(serving: ServingApp | None = None) -> FastAPI:
    serving = serving or ServingApp()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await serving.start()
        yield
        await serving.stop()

    app = FastAPI(title="coalmine", docs_url="/docs", lifespan=lifespan)
    app.state.serving = serving

    @app.post("/serve")
    async def serve(body: ServeRequest) -> dict:
        return await app.state.serving.serve(body.query_id)

    @app.get("/state")
    async def state() -> dict:
        return app.state.serving.state()

    @app.get("/events/stream")
    async def events_stream(max_seconds: float | None = None) -> StreamingResponse:
        # max_seconds bounds the stream for tests and curl demos; the
        # dashboard's EventSource omits it and tails forever.
        async def generate():
            yield ": connected\n\n"
            cursor = 0
            started = time.monotonic()
            while max_seconds is None or time.monotonic() - started < max_seconds:
                rows = app.state.serving.store.tail(cursor, limit=200)
                for rowid, event in rows:
                    cursor = rowid
                    yield (
                        "data: "
                        + json.dumps({"type": event.type, "seq": event.seq, **event.payload})
                        + "\n\n"
                    )
                await asyncio.sleep(0.5)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/")
    async def dashboard() -> FileResponse:
        return FileResponse(Path(__file__).parent / "dashboard.html")

    return app


def main() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=8123, log_level="warning")


if __name__ == "__main__":
    main()
