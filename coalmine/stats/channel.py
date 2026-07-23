"""Judge-as-noisy-channel model.

The decision engine never observes ground truth — it observes verdicts from an
LLM judge with imperfect accuracy and a nonzero tie rate. This module maps
between the true challenger win rate and the win rate seen through the judge,
so sequential tests can be designed on the scale they actually observe.

A true win-rate shift of d reaches the tests attenuated to d * (2a - 1), where
a is judge accuracy. At a = 0.85 a 5-point effect arrives as a 3.5-point one;
tests designed on the true scale would stop mysteriously later than theory
predicts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeChannel:
    accuracy: float = 0.85  # P(judge picks the truly better response | decisive verdict)
    tie_rate: float = 0.10  # P(judge declares a tie); tests consume decisive verdicts only

    def __post_init__(self) -> None:
        if not 0.5 < self.accuracy <= 1.0:
            raise ValueError("accuracy must be in (0.5, 1.0] — at 0.5 the judge carries no signal")
        if not 0.0 <= self.tie_rate < 1.0:
            raise ValueError("tie_rate must be in [0, 1)")

    def observed_win_rate(self, true_win_rate: float) -> float:
        """True challenger win rate -> win rate seen in decisive judge verdicts."""
        a = self.accuracy
        return true_win_rate * a + (1.0 - true_win_rate) * (1.0 - a)

    def true_win_rate(self, observed_win_rate: float) -> float:
        """Invert the channel: observed decisive win rate -> true win rate."""
        a = self.accuracy
        return (observed_win_rate - (1.0 - a)) / (2.0 * a - 1.0)

    def attenuation(self) -> float:
        """A true shift of d appears as a shift of d * attenuation() through the judge."""
        return 2.0 * self.accuracy - 1.0

    def requests_per_verdict(self, sampling_rate: float) -> float:
        """Expected live requests consumed per decisive verdict at a given judge sampling rate."""
        if not 0.0 < sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be in (0, 1]")
        return 1.0 / (sampling_rate * (1.0 - self.tie_rate))
