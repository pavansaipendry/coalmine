import numpy as np
import pytest

from coalmine.core.events import Event
from coalmine.core.replay import content_fingerprint, measured_epsilon, rolling_epsilon


def _shadow(seq: int, index: int, bad: bool, config: str = "challenger") -> Event:
    return Event(
        "r",
        seq,
        "shadow_response",
        {"request_index": index, "config_id": config, "source_pool": "bad" if bad else "good"},
        0.0,
    )


def test_fingerprint_ignores_event_order_and_run_identity():
    a = [
        Event("run-a", 0, "request_received", {"request_index": 0, "topic": "t"}, 1.0),
        Event("run-a", 1, "champion_response", {"request_index": 0, "text": "x"}, 2.0),
    ]
    b = [
        Event("run-b", 5, "champion_response", {"request_index": 0, "text": "x"}, 9.0),
        Event("run-b", 7, "request_received", {"request_index": 0, "topic": "t"}, 8.0),
    ]
    assert content_fingerprint(a) == content_fingerprint(b)


def test_fingerprint_sensitive_to_content():
    a = [Event("r", 0, "champion_response", {"request_index": 0, "text": "x"}, 1.0)]
    b = [Event("r", 0, "champion_response", {"request_index": 0, "text": "y"}, 1.0)]
    assert content_fingerprint(a) != content_fingerprint(b)


def test_measured_epsilon_window():
    events = [_shadow(i, i, bad=(i >= 50)) for i in range(100)]
    assert measured_epsilon(events, "challenger", 0, 50) == 0.0
    assert measured_epsilon(events, "challenger", 50, 100) == 1.0
    with pytest.raises(ValueError):
        measured_epsilon(events, "other-config", 0, 100)


def test_rolling_epsilon_tracks_step():
    events = [_shadow(i, i, bad=(i >= 500)) for i in range(1000)]
    centers, rolled = rolling_epsilon(events, "challenger", window=100)
    assert rolled[0] == 0.0
    assert rolled[-1] == pytest.approx(1.0)
    mid = np.searchsorted(centers, 500)
    assert 0.0 < rolled[mid] < 1.0
