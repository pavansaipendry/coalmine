"""Shadow router: the champion serves the user; challengers see the same
traffic silently.

The user path awaits only the champion. Shadow calls are fired as independent
tasks that are never awaited on the request path — a slow (or hung) challenger
cannot add a millisecond of user-facing latency. Whether a request is shadowed
is a seeded per-request draw, so the sampled set is identical across runs that
share a seed, which is what makes shadow-rate ablations paired comparisons.
"""

import asyncio

from coalmine.core.events import CHAMPION_RESPONSE, REQUEST_RECEIVED, SHADOW_RESPONSE
from coalmine.core.store import EventLog
from coalmine.serving.backends import Request, Response, ResponseBackend, request_rng


class ShadowRouter:
    def __init__(
        self,
        champion: ResponseBackend,
        challengers: list[ResponseBackend],
        shadow_rate: float,
        seed: int,
        log: EventLog,
    ):
        if not 0.0 <= shadow_rate <= 1.0:
            raise ValueError("shadow_rate must be in [0, 1]")
        self.champion = champion
        self.challengers = challengers
        self.shadow_rate = shadow_rate
        self.seed = seed
        self.log = log
        self._shadow_tasks: set[asyncio.Task] = set()

    async def route(self, request: Request) -> Response:
        self.log.emit(
            REQUEST_RECEIVED,
            {
                "request_index": request.index,
                "query_id": request.query.id,
                "query_text": request.query.text,
                "topic": request.query.topic,
            },
        )
        sampled = request_rng(self.seed, request.index, "__router__").random() < self.shadow_rate
        if sampled:
            for challenger in self.challengers:
                task = asyncio.create_task(self._shadow(challenger, request))
                self._shadow_tasks.add(task)
                task.add_done_callback(self._shadow_tasks.discard)
        response = await self.champion.generate(request)
        self.log.emit(CHAMPION_RESPONSE, self._response_payload(request, response))
        return response

    async def _shadow(self, backend: ResponseBackend, request: Request) -> None:
        response = await backend.generate(request)
        self.log.emit(SHADOW_RESPONSE, self._response_payload(request, response))

    async def drain(self) -> None:
        """Wait for all in-flight shadow calls — call before closing the log."""
        if self._shadow_tasks:
            await asyncio.gather(*list(self._shadow_tasks))

    @staticmethod
    def _response_payload(request: Request, response: Response) -> dict:
        return {
            "request_index": request.index,
            "query_id": request.query.id,
            "config_id": response.config_id,
            "source_pool": response.source_pool,
            "latency_ms": response.latency_ms,
            "text": response.text,
            **response.meta,
        }
