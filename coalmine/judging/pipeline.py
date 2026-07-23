"""Judging pipeline: turn a logged run's response pairs into verdict events.

Reads champion/shadow pairs from the event log, samples a seeded fraction,
runs each sampled pair through a position-randomized comparison, and appends
verdict_recorded events to the SAME run's stream (seq continues after the
existing events). The log stays the single source of truth: traffic, verdicts,
and — later — decisions all live in one replayable stream.

Judging a run twice with the same judge_id appends duplicate verdicts; give
each pass a distinct judge_id (metrics filter on it).
"""

import time

from coalmine.core.events import (
    CHAMPION_RESPONSE,
    REQUEST_RECEIVED,
    SHADOW_RESPONSE,
    VERDICT_RECORDED,
    Event,
)
from coalmine.core.store import EventStore
from coalmine.judging.judges import PairwiseJudge, compare_randomized
from coalmine.serving.backends import request_rng


class JudgePipeline:
    def __init__(
        self,
        judge: PairwiseJudge,
        seed: int,
        sampling_rate: float = 1.0,
        judge_id: str | None = None,
    ):
        if not 0.0 < sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be in (0, 1]")
        self.judge = judge
        self.seed = seed
        self.sampling_rate = sampling_rate
        self.judge_id = judge_id or judge.judge_id

    async def judge_run(
        self,
        store: EventStore,
        run_id: str,
        config_a: str = "champion",
        config_b: str = "challenger",
        randomize: bool = True,
        batch_size: int = 500,
    ) -> list[dict]:
        events = store.read(run_id)
        queries = {
            e.payload["request_index"]: e.payload["query_text"]
            for e in events
            if e.type == REQUEST_RECEIVED
        }
        a_responses = {
            e.payload["request_index"]: e.payload
            for e in events
            if e.type == CHAMPION_RESPONSE and e.payload["config_id"] == config_a
        }
        b_responses = {
            e.payload["request_index"]: e.payload
            for e in events
            if e.type == SHADOW_RESPONSE and e.payload["config_id"] == config_b
        }

        seq = store.max_seq(run_id) + 1
        verdicts: list[dict] = []
        batch: list[Event] = []
        for index in sorted(a_responses.keys() & b_responses.keys()):
            sampler = request_rng(self.seed, index, f"__judge_sample__{self.judge_id}")
            if sampler.random() >= self.sampling_rate:
                continue
            rng = request_rng(self.seed, index, f"__judge__{self.judge_id}")
            a, b = a_responses[index], b_responses[index]
            result = await compare_randomized(
                self.judge,
                queries[index],
                {"text": a["text"], "source_pool": a["source_pool"]},
                {"text": b["text"], "source_pool": b["source_pool"]},
                rng,
                randomize=randomize,
            )
            payload = {
                "request_index": index,
                "config_a": config_a,
                "config_b": config_b,
                "winner": result.winner,
                "first_shown": result.first_shown,
                "judge_id": self.judge_id,
            }
            verdicts.append(payload)
            batch.append(Event(run_id, seq, VERDICT_RECORDED, payload, time.time()))
            seq += 1
            if len(batch) >= batch_size:
                store.append_many(batch)
                batch = []
        if batch:
            store.append_many(batch)
        return verdicts
