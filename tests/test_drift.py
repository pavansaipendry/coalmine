import numpy as np
import pytest

from coalmine.fleet.drift import InputDriftMonitor, calibrate_psi_threshold, psi


def test_psi_zero_for_identical_distributions():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(p, p) == pytest.approx(0.0, abs=1e-12)
    assert psi(p, np.array([0.7, 0.1, 0.1, 0.1])) > 0.3


def _feed(monitor, topics_stream):
    for i, t in enumerate(topics_stream):
        if monitor.observe(t):
            return i + 1
    return None


def test_no_alarm_on_stationary_traffic_seeded():
    rng = np.random.default_rng(7)
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    topics = ["a", "b", "c", "d"]
    threshold = calibrate_psi_threshold(probs, 1_000, 100, 7_000, 0.05, trials=400, seed=1)
    monitor = InputDriftMonitor(topics, window=1_000, stride=100, threshold=threshold)
    stream = [topics[i] for i in rng.choice(4, size=8_000, p=probs)]
    assert _feed(monitor, stream) is None


def test_detects_topic_shift_quickly_seeded():
    rng = np.random.default_rng(9)
    topics = ["a", "b", "c", "d"]
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    shifted = np.array([0.70, 0.10, 0.10, 0.10])
    threshold = calibrate_psi_threshold(uniform, 1_000, 100, 6_000, 0.05, trials=400, seed=2)
    monitor = InputDriftMonitor(topics, window=1_000, stride=100, threshold=threshold)
    stream = [topics[i] for i in rng.choice(4, size=4_000, p=uniform)]
    stream += [topics[i] for i in rng.choice(4, size=3_000, p=shifted)]
    alarm_at = _feed(monitor, stream)
    assert alarm_at is not None
    delay = alarm_at - 4_000
    assert 0 < delay < 1_500, f"shift detected {delay} requests after it began"


def test_reference_freezes_after_first_window():
    monitor = InputDriftMonitor(["a", "b"], window=100, stride=50, threshold=0.1)
    for _ in range(50):
        monitor.observe("a")
    for _ in range(50):
        monitor.observe("b")
    assert monitor._reference is not None
    assert monitor._reference[0] == pytest.approx(0.5)
