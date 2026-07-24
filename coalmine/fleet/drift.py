"""Input drift monitor: is the traffic still the traffic the tests assume?

Every sequential test in the decision engine assumes an iid verdict stream;
a shift in the query mix silently breaks that assumption, and — when config
quality differs across topics — moves the aggregate win rate with no change
in either config (the mix-shift false alarm the drift gate exists to
prevent). The monitor watches the topic distribution in a sliding window
against a frozen reference window and alarms on population-stability-index
(PSI) excursions.

Same discipline as every detector here: the threshold is calibrated by
simulation to a false-alarm budget over the monitoring horizon — repeated
window checks have exactly the peeking pathology of repeated t-tests, so the
budget is over the max statistic, not per check. Topic labels stand in for
embedding clusters until real datasets arrive.
"""

from dataclasses import dataclass, field

import numpy as np


def psi(reference: np.ndarray, window: np.ndarray, floor: float = 1e-4) -> float:
    """Population stability index between two categorical distributions."""
    p = np.maximum(np.asarray(reference, dtype=float), floor)
    q = np.maximum(np.asarray(window, dtype=float), floor)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def calibrate_psi_threshold(
    probs: np.ndarray,
    window: int,
    stride: int,
    n_requests: int,
    false_alarm_rate: float,
    trials: int = 500,
    seed: int = 0,
) -> float:
    """Threshold = (1 - rate) quantile of the max PSI under stationary traffic,
    simulating the monitor's OWN procedure: an estimated (noisy) reference from
    the first window, then overlapping sliding windows every `stride` over a
    null stream of n_requests. Calibrating against the true reference with
    independent windows understates the max statistic — the deterministic test
    suite caught exactly that miscalibration."""
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs, dtype=float)
    k = len(probs)
    checks = np.arange(window, n_requests + 1, stride)
    max_psi = np.zeros(trials)
    for t in range(trials):
        reference = rng.multinomial(window, probs) / window
        stream = rng.choice(k, size=n_requests, p=probs)
        onehot = np.zeros((n_requests + 1, k))
        onehot[np.arange(1, n_requests + 1), stream] = 1.0
        cum = np.cumsum(onehot, axis=0)
        window_counts = cum[checks] - cum[checks - window]
        q = np.maximum(window_counts / window, 1e-4)
        q = q / q.sum(axis=1, keepdims=True)
        p = np.maximum(reference, 1e-4)
        p = p / p.sum()
        stats = np.sum((q - p) * np.log(q / p), axis=1)
        max_psi[t] = stats.max()
    return float(np.quantile(max_psi, 1.0 - false_alarm_rate))


@dataclass
class InputDriftMonitor:
    topics: list[str]
    window: int = 1_000
    stride: int = 100
    threshold: float = 0.05  # calibrate with calibrate_psi_threshold

    _reference: np.ndarray | None = field(init=False, default=None)
    _counts: dict[str, int] = field(init=False, default_factory=dict)
    _buffer: list[str] = field(init=False, default_factory=list)
    n_seen: int = field(init=False, default=0)
    alarmed_at: int | None = field(init=False, default=None)
    last_psi: float = field(init=False, default=0.0)

    def observe(self, topic: str) -> bool:
        """Feed one request's topic. Returns True once drift has alarmed.

        The first `window` requests freeze the reference distribution; after
        that a sliding window is scored every `stride` requests.
        """
        if self.alarmed_at is not None:
            return True
        self.n_seen += 1
        self._buffer.append(topic)
        if self._reference is None:
            if self.n_seen == self.window:
                counts = np.array([self._buffer.count(t) for t in self.topics], dtype=float)
                self._reference = counts / counts.sum()
                self._buffer = []
            return False
        if len(self._buffer) > self.window:
            self._buffer = self._buffer[-self.window :]
        due = len(self._buffer) == self.window and (self.n_seen % self.stride == 0)
        if due:
            counts = np.array([self._buffer.count(t) for t in self.topics], dtype=float)
            self.last_psi = psi(self._reference, counts / counts.sum())
            if self.last_psi >= self.threshold:
                self.alarmed_at = self.n_seen
                return True
        return False
