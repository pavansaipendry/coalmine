import os
import uuid

import pytest

from coalmine.core.events import Event
from coalmine.core.store import EventLog, PostgresEventStore, SQLiteEventStore


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "sqlite":
        s = SQLiteEventStore(tmp_path / "events.db")
    else:
        dsn = os.environ.get("COALMINE_PG_DSN")
        if not dsn:
            pytest.skip("COALMINE_PG_DSN not set")
        s = PostgresEventStore(dsn)
    yield s
    s.close()


def _run_id() -> str:
    return f"test-{uuid.uuid4().hex[:12]}"


def _events(run_id: str, n: int) -> list[Event]:
    return [Event(run_id, i, "request_received", {"i": i}, 1000.0 + i) for i in range(n)]


def test_append_read_roundtrip(store):
    run = _run_id()
    store.append_many(_events(run, 5))
    got = store.read(run)
    assert [e.seq for e in got] == list(range(5))
    assert got[3].payload == {"i": 3}
    assert got[0].type == "request_received"


def test_idempotent_redelivery(store):
    # At-least-once delivery upstream must yield exactly-once storage.
    run = _run_id()
    batch = _events(run, 8)
    store.append_many(batch)
    store.append_many(batch)
    store.append_many(batch[3:6])
    assert len(store.read(run)) == 8


def test_run_isolation(store):
    a, b = _run_id(), _run_id()
    store.append_many(_events(a, 4))
    store.append_many(_events(b, 2))
    assert len(store.read(a)) == 4
    assert len(store.read(b)) == 2
    assert {a, b} <= set(store.run_ids())


def test_read_sorted_regardless_of_append_order(store):
    run = _run_id()
    events = _events(run, 6)
    store.append_many(events[3:])
    store.append_many(events[:3])
    assert [e.seq for e in store.read(run)] == list(range(6))


async def test_event_log_batches_counts_and_drains(tmp_path):
    s = SQLiteEventStore(tmp_path / "log.db")
    log = EventLog(s, "logrun", batch_size=64, flush_interval=0.05)
    await log.start()
    for i in range(1000):
        log.emit("request_received", {"i": i})
    await log.close()
    got = s.read("logrun")
    assert len(got) == 1000
    assert [e.seq for e in got] == list(range(1000))
    assert log.counts["request_received"] == 1000
    s.close()
