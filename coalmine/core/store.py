"""Event stores: SQLite (zero-setup dev/CI) and Postgres (the fleet backend).

Both expose the same tiny sync interface behind EventLog's async batching
writer. Appends are idempotent — (run_id, seq) is the primary key and
redelivered batches are dropped on conflict — so at-least-once delivery from
upstream yields exactly-once storage, which is what makes the later chaos
testing (kill a worker mid-batch, redeliver) safe by construction.
"""

import asyncio
import itertools
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Protocol, Sequence

import psycopg
from psycopg.types.json import Jsonb

from coalmine.core.events import Event


class EventStore(Protocol):
    def append_many(self, events: Sequence[Event]) -> None: ...

    def read(self, run_id: str) -> list[Event]: ...

    def run_ids(self) -> list[str]: ...

    def close(self) -> None: ...


class SQLiteEventStore:
    def __init__(self, path: str | Path):
        # check_same_thread=False: EventLog flushes via to_thread; access is
        # serialized by the single writer task, never concurrent.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            " run_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL,"
            " type TEXT NOT NULL, payload TEXT NOT NULL,"
            " PRIMARY KEY (run_id, seq))"
        )
        self.conn.commit()

    def append_many(self, events: Sequence[Event]) -> None:
        self.conn.executemany(
            "INSERT OR IGNORE INTO events (run_id, seq, ts, type, payload) VALUES (?,?,?,?,?)",
            [(e.run_id, e.seq, e.ts, e.type, json.dumps(e.payload)) for e in events],
        )
        self.conn.commit()

    def read(self, run_id: str) -> list[Event]:
        rows = self.conn.execute(
            "SELECT run_id, seq, ts, type, payload FROM events WHERE run_id=? ORDER BY seq",
            (run_id,),
        ).fetchall()
        return [Event(r[0], r[1], r[3], json.loads(r[4]), r[2]) for r in rows]

    def run_ids(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT run_id FROM events ORDER BY run_id").fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self.conn.close()


class PostgresEventStore:
    def __init__(self, dsn: str):
        self.conn = psycopg.connect(dsn)
        with self.conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                " run_id TEXT NOT NULL, seq BIGINT NOT NULL, ts DOUBLE PRECISION NOT NULL,"
                " type TEXT NOT NULL, payload JSONB NOT NULL,"
                " PRIMARY KEY (run_id, seq))"
            )
        self.conn.commit()

    def append_many(self, events: Sequence[Event]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO events (run_id, seq, ts, type, payload) VALUES (%s,%s,%s,%s,%s)"
                " ON CONFLICT (run_id, seq) DO NOTHING",
                [(e.run_id, e.seq, e.ts, e.type, Jsonb(e.payload)) for e in events],
            )
        self.conn.commit()

    def read(self, run_id: str) -> list[Event]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT run_id, seq, ts, type, payload FROM events WHERE run_id=%s ORDER BY seq",
                (run_id,),
            )
            rows = cur.fetchall()
        return [Event(r[0], r[1], r[3], r[4], r[2]) for r in rows]

    def run_ids(self) -> list[str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT DISTINCT run_id FROM events ORDER BY run_id")
            return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()


_CLOSE = object()


class EventLog:
    """Async front door to a store: emit() is non-blocking; a single writer
    task drains the queue and flushes batches off the event loop, so logging
    never sits on the user-facing request path."""

    def __init__(
        self,
        store: EventStore,
        run_id: str,
        batch_size: int = 256,
        flush_interval: float = 0.25,
    ):
        self.store = store
        self.run_id = run_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.counts: dict[str, int] = defaultdict(int)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._seq = itertools.count()
        self._task: asyncio.Task | None = None

    def emit(self, type_: str, payload: dict) -> None:
        self.counts[type_] += 1
        self._queue.put_nowait(Event(self.run_id, next(self._seq), type_, payload, time.time()))

    async def start(self) -> None:
        self._task = asyncio.create_task(self._writer())

    async def close(self) -> None:
        self._queue.put_nowait(_CLOSE)
        if self._task is not None:
            await self._task

    async def _writer(self) -> None:
        buf: list[Event] = []
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self.flush_interval)
            except TimeoutError:
                await self._flush(buf)
                continue
            if item is _CLOSE:
                break
            buf.append(item)
            if len(buf) >= self.batch_size:
                await self._flush(buf)
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if item is not _CLOSE:
                buf.append(item)
        await self._flush(buf)

    async def _flush(self, buf: list[Event]) -> None:
        if buf:
            batch = list(buf)
            buf.clear()
            await asyncio.to_thread(self.store.append_many, batch)
