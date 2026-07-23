"""Immutable events — the system's source of truth.

Everything that happens (a request arrives, a config responds, a verdict
lands, a decision fires) is an append-only event. Detectors are then
backtestable: feed recorded history through a new engine version and compare.

Determinism contract: response *content* is a pure function of
(seed, request_index, config), so payloads are identical across same-seed
runs. Event *ordering* for concurrent shadow calls is scheduler-dependent and
deliberately not part of the contract — replay comparisons canonicalize by
sorting payloads (see coalmine.core.replay).
"""

from dataclasses import dataclass

REQUEST_RECEIVED = "request_received"
CHAMPION_RESPONSE = "champion_response"
SHADOW_RESPONSE = "shadow_response"
RUN_STARTED = "run_started"
RUN_FINISHED = "run_finished"


@dataclass(frozen=True)
class Event:
    run_id: str
    seq: int  # unique, monotone per run — assigned at emit time
    type: str
    payload: dict
    ts: float  # wall clock, informational only; never part of determinism checks
