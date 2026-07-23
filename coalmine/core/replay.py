"""Replay utilities: recover ground truth and verify determinism from the log.

The event log is the source of truth, so claims about a run ("we injected a
10% regression at request 10,000") must be recoverable from it — these helpers
do that recovery. Fingerprints canonicalize away scheduler-dependent event
ordering: content is deterministic, interleaving is not.
"""

import hashlib
import json

import numpy as np

from coalmine.core.events import CHAMPION_RESPONSE, REQUEST_RECEIVED, SHADOW_RESPONSE, Event

DATA_EVENTS = {REQUEST_RECEIVED, CHAMPION_RESPONSE, SHADOW_RESPONSE}


def content_fingerprint(events: list[Event]) -> str:
    """Order-free hash of a run's request/response content.

    Two same-seed runs must produce equal fingerprints regardless of how the
    scheduler interleaved their shadow tasks. run_started/run_finished are
    excluded: they carry run identity and wall-clock timing.
    """
    parts = sorted(
        json.dumps({"type": e.type, **e.payload}, sort_keys=True)
        for e in events
        if e.type in DATA_EVENTS
    )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def shadow_flags(events: list[Event], config_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(request_index, served_from_bad_pool) arrays for one config's shadow
    responses, sorted by request index."""
    rows = sorted(
        (e.payload["request_index"], e.payload["source_pool"] == "bad")
        for e in events
        if e.type == SHADOW_RESPONSE and e.payload["config_id"] == config_id
    )
    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=bool)
    idx, bad = zip(*rows)
    return np.array(idx, dtype=np.int64), np.array(bad, dtype=bool)


def measured_epsilon(
    events: list[Event], config_id: str, start_index: int, end_index: int
) -> float:
    """Fraction of shadow responses served from the bad pool in an index range —
    the recovered value of the injected ε."""
    idx, bad = shadow_flags(events, config_id)
    mask = (idx >= start_index) & (idx < end_index)
    if not mask.any():
        raise ValueError(f"no shadow responses for {config_id} in [{start_index}, {end_index})")
    return float(bad[mask].mean())


def rolling_epsilon(
    events: list[Event], config_id: str, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean of the bad-pool fraction over shadow responses, for
    plotting recovered ε against the configured step. Returns (window-center
    request indices, rolling ε)."""
    idx, bad = shadow_flags(events, config_id)
    if len(idx) < window:
        raise ValueError(f"need at least {window} shadow responses, have {len(idx)}")
    kernel = np.ones(window) / window
    rolled = np.convolve(bad.astype(float), kernel, mode="valid")
    centers = idx[window // 2 : window // 2 + len(rolled)]
    return centers, rolled
