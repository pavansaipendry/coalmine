"""Bus semantics, identical across implementations: group load-balancing,
ack, and reclaim of a dead consumer's pending messages. The Redis tests run
against a live server (local or CI service container) under an isolated,
cleaned-up key prefix; they skip if no server is reachable."""

import asyncio
import os
import uuid

import pytest

from coalmine.fleet.bus import InMemoryBus, RedisStreamBus


def _redis_url() -> str:
    return os.environ.get("COALMINE_REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture(params=["memory", "redis"])
async def bus(request):
    if request.param == "memory":
        yield InMemoryBus()
        return
    b = RedisStreamBus(_redis_url(), prefix=f"coalmine:test:{uuid.uuid4().hex[:12]}")
    try:
        await b.redis.ping()
    except Exception:
        await b.close()
        pytest.skip("no reachable Redis")
    yield b
    await b.cleanup()
    await b.close()


async def _drain(bus, streams, group, consumer, n, ack=True, timeout=5.0):
    got = []

    async def consume():
        async for message in bus.consume(streams, group, consumer, reclaim_idle_ms=300):
            got.append(message)
            if ack:
                await bus.ack(message, group)
            if len(got) >= n:
                return

    await asyncio.wait_for(consume(), timeout)
    return got


async def test_publish_consume_ack(bus):
    for i in range(5):
        await bus.publish("s1", {"i": i})
    got = await _drain(bus, ["s1"], "g", "c0", 5)
    assert [m.payload["i"] for m in got] == [0, 1, 2, 3, 4]
    assert await bus.pending_count("s1", "g") == 0


async def test_group_shares_work_exactly_once(bus):
    # Consumer "a" processes 5 and leaves; "b" must end up with exactly the
    # other 25 (via the shared group cursor, plus reclaim of anything "a"
    # prefetched but never handled). Union exact, intersection empty.
    for i in range(30):
        await bus.publish("s2", {"i": i})
    got_a = await _drain(bus, ["s2"], "g", "a", 5)
    await asyncio.sleep(0.4)  # let a's abandoned prefetch become reclaimable
    got_b = await _drain(bus, ["s2"], "g", "b", 25)
    a_ids = {m.payload["i"] for m in got_a}
    b_ids = {m.payload["i"] for m in got_b}
    assert len(a_ids) == 5 and len(b_ids) == 25
    assert a_ids | b_ids == set(range(30))
    assert not (a_ids & b_ids), "an acked message must never be redelivered"


async def test_dead_consumer_messages_are_reclaimed(bus):
    for i in range(6):
        await bus.publish("s3", {"i": i})
    # Consumer "dead" handles 3 messages without acking, then vanishes. (The
    # Redis consumer may have prefetched more into its pending list — that is
    # exactly the crash mode reclaim exists for.)
    dead_got = await _drain(bus, ["s3"], "g", "dead", 3, ack=False)
    assert len(dead_got) == 3
    assert await bus.pending_count("s3", "g") >= 3
    await asyncio.sleep(0.4)  # exceed reclaim_idle_ms
    # A healthy consumer must receive all 6 — reclaimed plus new — exactly once.
    got = await _drain(bus, ["s3"], "g", "alive", 6)
    assert sorted(m.payload["i"] for m in got) == [0, 1, 2, 3, 4, 5]
    assert await bus.pending_count("s3", "g") == 0


async def test_multi_stream_consume(bus):
    await bus.publish("sa", {"from": "a"})
    await bus.publish("sb", {"from": "b"})
    got = await _drain(bus, ["sa", "sb"], "g", "c", 2)
    assert {m.payload["from"] for m in got} == {"a", "b"}


async def test_independent_groups_each_see_everything(bus):
    for i in range(4):
        await bus.publish("s4", {"i": i})
    g1 = await _drain(bus, ["s4"], "g1", "c", 4)
    g2 = await _drain(bus, ["s4"], "g2", "c", 4)
    assert [m.payload["i"] for m in g1] == [m.payload["i"] for m in g2] == [0, 1, 2, 3]
