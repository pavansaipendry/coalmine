"""Response backends: pool-served model configs and the ε-mixture wrapper.

Every random draw (response choice, latency, mixture coin) comes from an RNG
keyed by (seed, request_index, stream name), never from a shared generator —
so response content is a pure function of the request regardless of how
concurrent tasks get scheduled. That per-request keying is what makes
same-seed runs byte-identical in content while shadow tasks interleave freely.

The ε-mixture is the regression injector: with probability ε serve from a
deliberately-bad response pool, so the ground-truth effect size is exactly ε
by construction — the property that makes detection-latency claims
falsifiable. source_pool is recorded per response; it is simulation-only
ground truth (a real deployment has no such label — the judge estimates it).
"""

import asyncio
import math
import zlib
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from coalmine.traffic.dataset import Query
from coalmine.traffic.pools import ResponsePool


def request_rng(seed: int, request_index: int, stream: str) -> np.random.Generator:
    """Independent RNG stream per (seed, request, purpose)."""
    return np.random.default_rng([seed, request_index, zlib.crc32(stream.encode())])


@dataclass(frozen=True)
class Request:
    index: int
    query: Query


@dataclass(frozen=True)
class Response:
    config_id: str
    text: str
    source_pool: str  # "good" | "bad" — simulation-only ground truth
    latency_ms: float
    meta: dict = field(default_factory=dict)


class LatencyModel(Protocol):
    def sample(self, rng: np.random.Generator) -> float: ...


@dataclass(frozen=True)
class LogNormalLatency:
    median_ms: float
    sigma: float

    def sample(self, rng: np.random.Generator) -> float:
        return self.median_ms * math.exp(self.sigma * rng.standard_normal())


@dataclass(frozen=True)
class FixedLatency:
    ms: float

    def sample(self, rng: np.random.Generator) -> float:
        return self.ms


class ResponseBackend(Protocol):
    config_id: str

    async def generate(self, request: Request) -> Response: ...


class PoolBackend:
    """Serves pre-generated responses for one config from one quality pool."""

    def __init__(
        self,
        config_id: str,
        pool: ResponsePool,
        quality: str,
        seed: int,
        latency: LatencyModel | None = None,
    ):
        self.config_id = config_id
        self.pool = pool
        self.quality = quality
        self.seed = seed
        self.latency = latency

    async def generate(self, request: Request) -> Response:
        rng = request_rng(self.seed, request.index, f"{self.config_id}:{self.quality}")
        choices = self.pool.responses(request.query.id, self.quality)
        text = choices[int(rng.integers(len(choices)))]
        latency_ms = 0.0
        if self.latency is not None:
            latency_ms = round(self.latency.sample(rng), 3)
            await asyncio.sleep(latency_ms / 1000.0)
        return Response(self.config_id, text, self.quality, latency_ms)


def step_schedule(epsilon: float, changepoint: int | None) -> Callable[[Request], float]:
    """ε as a function of the request: 0 before the changepoint, ε after.
    changepoint=None means never regressed."""

    def schedule(request: Request) -> float:
        if changepoint is None or request.index < changepoint:
            return 0.0
        return epsilon

    return schedule


def topic_step_schedule(
    epsilon: float, changepoint: int, topic: str
) -> Callable[[Request], float]:
    """A regression confined to one topic — the stratified-monitoring scenario:
    aggregate quality barely moves while one slice of traffic degrades hard."""

    def schedule(request: Request) -> float:
        if request.query.topic != topic or request.index < changepoint:
            return 0.0
        return epsilon

    return schedule


class MixtureBackend:
    """One config that serves good responses except with probability
    ε(request_index), when it serves from the bad pool."""

    def __init__(
        self,
        config_id: str,
        good: PoolBackend,
        bad: PoolBackend,
        epsilon_schedule: Callable[[Request], float],
        seed: int,
    ):
        if good.config_id != config_id or bad.config_id != config_id:
            raise ValueError("inner pools must carry the mixture's config_id")
        self.config_id = config_id
        self.good = good
        self.bad = bad
        self.epsilon_schedule = epsilon_schedule
        self.seed = seed

    async def generate(self, request: Request) -> Response:
        eps = self.epsilon_schedule(request)
        rng = request_rng(self.seed, request.index, f"{self.config_id}#mixture")
        backend = self.bad if rng.random() < eps else self.good
        resp = await backend.generate(request)
        return Response(
            resp.config_id,
            resp.text,
            resp.source_pool,
            resp.latency_ms,
            {"epsilon_active": eps},
        )
