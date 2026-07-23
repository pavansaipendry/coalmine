from coalmine.core.events import CHAMPION_RESPONSE, REQUEST_RECEIVED, SHADOW_RESPONSE
from coalmine.core.store import EventLog, SQLiteEventStore
from coalmine.serving.backends import FixedLatency, PoolBackend, Request, request_rng
from coalmine.serving.router import ShadowRouter
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool

SEED = 21
N = 20
SHADOW_RATE = 0.5


def expected_sampled_indices() -> set[int]:
    return {
        i
        for i in range(N)
        if request_rng(SEED, i, "__router__").random() < SHADOW_RATE
    }


async def test_shadow_sampling_and_user_path_isolation(tmp_path):
    store = SQLiteEventStore(tmp_path / "router.db")
    queries = synthetic_queries(8)
    pool = ResponsePool.build_synthetic(queries)
    log = EventLog(store, "router-test", flush_interval=0.05)
    await log.start()

    champion = PoolBackend("champion", pool, "good", SEED)  # instant
    challenger = PoolBackend("challenger", pool, "good", SEED, latency=FixedLatency(800.0))
    router = ShadowRouter(champion, [challenger], SHADOW_RATE, SEED, log)

    for i in range(N):
        await router.route(Request(i, queries[i % len(queries)]))

    # Every user-facing response has returned, yet no 800ms shadow call has
    # finished — the user path never waited on a challenger.
    assert log.counts[CHAMPION_RESPONSE] == N
    assert log.counts[SHADOW_RESPONSE] == 0

    await router.drain()
    await log.close()

    events = store.read("router-test")
    shadow_indices = {
        e.payload["request_index"] for e in events if e.type == SHADOW_RESPONSE
    }
    assert shadow_indices == expected_sampled_indices()
    assert sum(1 for e in events if e.type == REQUEST_RECEIVED) == N
    store.close()


async def test_shadow_rate_zero_and_one(tmp_path):
    store = SQLiteEventStore(tmp_path / "rates.db")
    queries = synthetic_queries(4)
    pool = ResponsePool.build_synthetic(queries)
    for rate, expected in ((0.0, 0), (1.0, 10)):
        log = EventLog(store, f"rate-{rate}", flush_interval=0.05)
        await log.start()
        router = ShadowRouter(
            PoolBackend("champion", pool, "good", 1),
            [PoolBackend("challenger", pool, "good", 1)],
            rate,
            1,
            log,
        )
        for i in range(10):
            await router.route(Request(i, queries[i % 4]))
        await router.drain()
        await log.close()
        assert log.counts[SHADOW_RESPONSE] == expected
    store.close()
