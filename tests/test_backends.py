import pytest

from coalmine.serving.backends import (
    FixedLatency,
    MixtureBackend,
    PoolBackend,
    Request,
    request_rng,
    step_schedule,
)
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool


@pytest.fixture
def queries():
    return synthetic_queries(16)


@pytest.fixture
def pool(queries):
    return ResponsePool.build_synthetic(queries, variants=4)


async def test_pool_backend_content_is_deterministic(queries, pool):
    a = PoolBackend("champion", pool, "good", seed=7)
    b = PoolBackend("champion", pool, "good", seed=7)
    for i in range(20):
        req = Request(i, queries[i % len(queries)])
        assert (await a.generate(req)).text == (await b.generate(req)).text


async def test_different_seed_changes_content(queries, pool):
    a = PoolBackend("champion", pool, "good", seed=7)
    b = PoolBackend("champion", pool, "good", seed=8)
    texts_a = [(await a.generate(Request(i, queries[i % 16]))).text for i in range(30)]
    texts_b = [(await b.generate(Request(i, queries[i % 16]))).text for i in range(30)]
    assert texts_a != texts_b


async def test_zero_latency_without_model(queries, pool):
    backend = PoolBackend("champion", pool, "good", seed=1)
    resp = await backend.generate(Request(0, queries[0]))
    assert resp.latency_ms == 0.0
    assert resp.source_pool == "good"


async def test_fixed_latency_recorded(queries, pool):
    backend = PoolBackend("champion", pool, "good", seed=1, latency=FixedLatency(5.0))
    resp = await backend.generate(Request(0, queries[0]))
    assert resp.latency_ms == 5.0


async def test_mixture_respects_changepoint_exactly(queries, pool):
    mix = MixtureBackend(
        "challenger",
        PoolBackend("challenger", pool, "good", seed=3),
        PoolBackend("challenger", pool, "bad", seed=3),
        step_schedule(0.25, changepoint=500),
        seed=3,
    )
    pre = [await mix.generate(Request(i, queries[i % 16])) for i in range(0, 500, 7)]
    assert all(r.source_pool == "good" for r in pre)
    assert all(r.meta["epsilon_active"] == 0.0 for r in pre)

    post = [await mix.generate(Request(i, queries[i % 16])) for i in range(500, 2500)]
    bad_rate = sum(r.source_pool == "bad" for r in post) / len(post)
    assert abs(bad_rate - 0.25) < 0.03
    assert all(r.meta["epsilon_active"] == 0.25 for r in post)


async def test_mixture_draw_is_reproducible(queries, pool):
    def build():
        return MixtureBackend(
            "challenger",
            PoolBackend("challenger", pool, "good", seed=9),
            PoolBackend("challenger", pool, "bad", seed=9),
            step_schedule(0.5, changepoint=0),
            seed=9,
        )

    a, b = build(), build()
    for i in range(50):
        req = Request(i, queries[i % 16])
        assert (await a.generate(req)).source_pool == (await b.generate(req)).source_pool


def test_mixture_rejects_mismatched_config():
    queries = synthetic_queries(4)
    pool = ResponsePool.build_synthetic(queries)
    with pytest.raises(ValueError):
        MixtureBackend(
            "challenger",
            PoolBackend("other", pool, "good", seed=1),
            PoolBackend("challenger", pool, "bad", seed=1),
            step_schedule(0.1, 0),
            seed=1,
        )


def test_request_rng_streams_are_independent():
    a = request_rng(1, 10, "champion:good").random()
    b = request_rng(1, 10, "challenger:good").random()
    c = request_rng(1, 11, "champion:good").random()
    assert a != b and a != c
    assert request_rng(1, 10, "champion:good").random() == a
