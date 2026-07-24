import numpy as np
import pytest

from coalmine.sim.runner import run_msprt_batch
from coalmine.sim.verdicts import gen_wins
from coalmine.stats.multiarm import MultiArmMonitor

ARMS = ["arm-a", "arm-b", "arm-c", "arm-d"]


def test_correction_splits_alpha():
    corrected = MultiArmMonitor(ARMS, alpha=0.05)
    uncorrected = MultiArmMonitor(ARMS, alpha=0.05, corrected=False)
    assert corrected.per_arm_alpha == pytest.approx(0.0125)
    assert uncorrected.per_arm_alpha == pytest.approx(0.05)
    # A smaller per-arm alpha means a strictly higher rejection bar.
    assert (
        corrected.tests["arm-a"].log_threshold > uncorrected.tests["arm-a"].log_threshold
    )


def test_clearly_better_arm_promotes_once():
    monitor = MultiArmMonitor(ARMS, alpha=0.05)
    newly = 0
    for _ in range(500):
        newly += monitor.update("arm-b", True)
    assert monitor.promoted == ["arm-b"]
    assert newly == 1  # promotion reported exactly once
    assert not monitor.tests["arm-a"].rejected


def test_family_error_controlled_and_uncorrected_inflated_seeded():
    # 4 null arms, independent verdict streams: family-wise error compares
    # corrected (alpha/4 per arm) vs uncorrected (alpha per arm).
    rng = np.random.default_rng(53)
    trials, horizon, k = 600, 5_000, len(ARMS)
    rejected_corr = np.zeros(trials, dtype=bool)
    rejected_uncorr = np.zeros(trials, dtype=bool)
    for _ in range(k):
        wins = gen_wins(rng, trials, horizon, 0.5)
        rejected_corr |= run_msprt_batch(wins, p0=0.5, alpha=0.05 / k) > 0
        rejected_uncorr |= run_msprt_batch(wins, p0=0.5, alpha=0.05) > 0
    fwer_corr = float(rejected_corr.mean())
    fwer_uncorr = float(rejected_uncorr.mean())
    assert fwer_corr <= 0.05, f"corrected FWER {fwer_corr} exceeds budget"
    assert fwer_uncorr > fwer_corr, "correction should reduce family-wise error"


def test_invalid_arms():
    with pytest.raises(ValueError):
        MultiArmMonitor([])
    with pytest.raises(ValueError):
        MultiArmMonitor(["a", "a"])
