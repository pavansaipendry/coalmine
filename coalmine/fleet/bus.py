"""Message bus: Redis Streams with consumer groups, plus an in-memory twin.

Both implementations expose identical semantics — publish to a stream,
consume via a named group (messages are load-balanced across consumers in the
group), ack on completion, and reclaim messages left pending by a dead
consumer after an idle timeout. Exactly-once *processing* is not promised by
the bus (crashes cause redelivery); it is recovered downstream by the
idempotent event store: message handling is deterministic and event seqs are
derived from request identity, so a redelivered message writes the same
(run_id, seq) row and dedups. At-least-once delivery + idempotent effects =
converged final state — the property the chaos experiment kills workers to
prove.

InMemoryBus mirrors the semantics for zero-dependency tests and dev;
RedisStreamBus is verified against a live Redis locally and in CI.
"""

import asyncio
import itertools
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import redis.asyncio as aioredis


@dataclass(frozen=True)
class Message:
    stream: str
    message_id: str
    payload: dict


class Bus(Protocol):
    async def publish(self, stream: str, payload: dict) -> None: ...

    def consume(
        self, streams: list[str], group: str, consumer: str, reclaim_idle_ms: int = 1000
    ) -> AsyncIterator[Message]: ...

    async def ack(self, message: Message, group: str) -> None: ...

    async def close(self) -> None: ...


class InMemoryBus:
    """Single-process twin of the Redis bus: per-group cursors, pending lists,
    and idle-based reclaim, over plain asyncio structures."""

    def __init__(self):
        self._streams: dict[str, list[tuple[str, dict]]] = {}
        self._cursors: dict[tuple[str, str], int] = {}
        self._pending: dict[tuple[str, str], dict[str, tuple[str, float]]] = {}
        self._seq = itertools.count()
        self._lock = asyncio.Lock()

    async def publish(self, stream: str, payload: dict) -> None:
        async with self._lock:
            self._streams.setdefault(stream, []).append((f"m-{next(self._seq)}", payload))

    async def _next(
        self, streams: list[str], group: str, consumer: str, reclaim_idle_ms: int
    ) -> Message | None:
        async with self._lock:
            now = time.monotonic()
            for stream in streams:
                key = (stream, group)
                pending = self._pending.setdefault(key, {})
                for mid, (owner, since) in pending.items():
                    if (now - since) * 1000.0 >= reclaim_idle_ms:
                        pending[mid] = (consumer, now)
                        payload = dict(next(p for i, p in self._streams[stream] if i == mid))
                        return Message(stream, mid, payload)
                entries = self._streams.setdefault(stream, [])
                cursor = self._cursors.setdefault(key, 0)
                if cursor < len(entries):
                    mid, payload = entries[cursor]
                    self._cursors[key] = cursor + 1
                    pending[mid] = (consumer, now)
                    return Message(stream, mid, dict(payload))
        return None

    async def consume(
        self, streams: list[str], group: str, consumer: str, reclaim_idle_ms: int = 1000
    ) -> AsyncIterator[Message]:
        while True:
            message = await self._next(streams, group, consumer, reclaim_idle_ms)
            if message is None:
                await asyncio.sleep(0.005)
                continue
            yield message

    async def ack(self, message: Message, group: str) -> None:
        async with self._lock:
            self._pending.get((message.stream, group), {}).pop(message.message_id, None)

    async def pending_count(self, stream: str, group: str) -> int:
        async with self._lock:
            return len(self._pending.get((stream, group), {}))

    async def close(self) -> None:
        return None


class RedisStreamBus:
    """Redis Streams + consumer groups (XADD / XREADGROUP / XACK / XAUTOCLAIM).

    All keys are namespaced under a prefix so runs are isolated and cleanup is
    a prefix delete — the local Redis is shared with other projects.
    """

    def __init__(self, url: str, prefix: str):
        self.redis = aioredis.from_url(url, decode_responses=True)
        self.prefix = prefix
        self._groups_created: set[tuple[str, str]] = set()

    def _key(self, stream: str) -> str:
        return f"{self.prefix}:{stream}"

    async def publish(self, stream: str, payload: dict) -> None:
        await self.redis.xadd(self._key(stream), {"data": json.dumps(payload)})

    async def _ensure_group(self, stream: str, group: str) -> None:
        if (stream, group) in self._groups_created:
            return
        try:
            await self.redis.xgroup_create(self._key(stream), group, id="0", mkstream=True)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        self._groups_created.add((stream, group))

    async def consume(
        self, streams: list[str], group: str, consumer: str, reclaim_idle_ms: int = 1000
    ) -> AsyncIterator[Message]:
        for stream in streams:
            await self._ensure_group(stream, group)
        backlog: deque[Message] = deque()
        last_reclaim = 0.0
        while True:
            if backlog:
                yield backlog.popleft()
                continue
            # Periodically recover messages a dead consumer left pending.
            now = time.monotonic()
            if now - last_reclaim >= reclaim_idle_ms / 2000.0:
                last_reclaim = now
                for stream in streams:
                    claimed = await self.redis.xautoclaim(
                        self._key(stream), group, consumer, min_idle_time=reclaim_idle_ms, count=16
                    )
                    for message_id, fields in claimed[1]:
                        if fields:
                            backlog.append(Message(stream, message_id, json.loads(fields["data"])))
                if backlog:
                    continue
            response = await self.redis.xreadgroup(
                group,
                consumer,
                {self._key(s): ">" for s in streams},
                count=64,
                block=100,
            )
            for full_key, entries in response or []:
                stream = full_key.removeprefix(f"{self.prefix}:")
                for message_id, fields in entries:
                    backlog.append(Message(stream, message_id, json.loads(fields["data"])))

    async def ack(self, message: Message, group: str) -> None:
        await self.redis.xack(self._key(message.stream), group, message.message_id)

    async def pending_count(self, stream: str, group: str) -> int:
        await self._ensure_group(stream, group)
        info = await self.redis.xpending(self._key(stream), group)
        return int(info["pending"])

    async def cleanup(self) -> None:
        """Delete every key under this bus's prefix."""
        keys = [k async for k in self.redis.scan_iter(match=f"{self.prefix}:*")]
        if keys:
            await self.redis.delete(*keys)

    async def close(self) -> None:
        await self.redis.aclose()
