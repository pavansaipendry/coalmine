import numpy as np
import pytest

from coalmine.canary.lifecycle import (
    ADVANCED,
    PROMOTED,
    ROLLED_BACK,
    STAGES,
    CanaryLifecycle,
)


def _feed(lifecycle: CanaryLifecycle, wins) -> list:
    transitions = []
    for i, win in enumerate(wins):
        t = lifecycle.update(bool(win), i)
        if t:
            transitions.append(t)
        if lifecycle.terminal:
            break
    return transitions


def _stream(seed: int, n: int, w: float):
    return np.random.default_rng(seed).random(n) < w


def test_equal_challenger_promotes_through_every_stage():
    lifecycle = CanaryLifecycle(cusum_h=6.0, margin=0.10, alpha=0.2)
    transitions = _feed(lifecycle, _stream(3, 6_000, 0.5))
    kinds = [t.kind for t in transitions]
    assert kinds == [ADVANCED, ADVANCED, ADVANCED, PROMOTED]
    assert [t.stage for t in transitions[:-1]] == [name for name, _ in STAGES[1:]]
    assert lifecycle.state == "promoted"
    assert lifecycle.share == 1.0
    # Shares ramp monotonically: 1% -> 5% -> 25% -> 100%.
    assert [t.share for t in transitions] == [0.01, 0.05, 0.25, 1.0]


def test_regressed_challenger_rolls_back_not_promotes():
    lifecycle = CanaryLifecycle(cusum_h=6.0, margin=0.10, alpha=0.2)
    transitions = _feed(lifecycle, _stream(5, 6_000, 0.38))
    assert lifecycle.state == "rolled_back"
    assert lifecycle.share == 0.0
    assert transitions[-1].kind == ROLLED_BACK
    assert ADVANCED not in [t.kind for t in transitions]


def test_mid_ramp_regression_rolls_back_after_advancing():
    lifecycle = CanaryLifecycle(cusum_h=6.0, margin=0.10, alpha=0.2)
    healthy = _stream(7, 700, 0.5)  # long enough for 1-2 gates, not the full ramp
    regressed = _stream(8, 4_000, 0.35)
    transitions = _feed(lifecycle, np.concatenate([healthy, regressed]))
    kinds = [t.kind for t in transitions]
    assert kinds[-1] == ROLLED_BACK
    assert ADVANCED in kinds, "should have ramped at least one stage before the regression"
    assert lifecycle.share == 0.0


def test_terminal_states_ignore_updates():
    lifecycle = CanaryLifecycle(cusum_h=6.0, margin=0.10, alpha=0.2)
    _feed(lifecycle, _stream(3, 6_000, 0.5))
    assert lifecycle.state == "promoted"
    seen = lifecycle.verdicts_seen
    assert lifecycle.update(False, 999_999) is None
    assert lifecycle.verdicts_seen == seen


def test_lifecycle_is_deterministic():
    a = CanaryLifecycle(cusum_h=6.0, margin=0.10, alpha=0.2)
    b = CanaryLifecycle(cusum_h=6.0, margin=0.10, alpha=0.2)
    wins = _stream(11, 5_000, 0.5)
    assert _feed(a, wins) == _feed(b, wins)


def test_invalid_margin():
    with pytest.raises(ValueError):
        CanaryLifecycle(cusum_h=6.0, margin=0.0)
