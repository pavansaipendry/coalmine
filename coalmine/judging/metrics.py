"""Verdict metrics: win rates and position bias, recovered from the log."""

from coalmine.core.events import VERDICT_RECORDED, Event


def verdicts_from_events(events: list[Event], judge_id: str | None = None) -> list[dict]:
    """verdict_recorded payloads, sorted by request index; optionally one judge's."""
    rows = [
        e.payload
        for e in events
        if e.type == VERDICT_RECORDED
        and (judge_id is None or e.payload["judge_id"] == judge_id)
    ]
    return sorted(rows, key=lambda p: p["request_index"])


def decisive(verdicts: list[dict]) -> list[dict]:
    return [v for v in verdicts if v["winner"] != "tie"]


def challenger_win_rate(verdicts: list[dict]) -> float:
    """P(config_b wins | decisive) — the quantity the sequential tests watch."""
    d = decisive(verdicts)
    if not d:
        raise ValueError("no decisive verdicts")
    return sum(v["winner"] == "b" for v in d) / len(d)


def challenger_wins(verdicts: list[dict]) -> list[bool]:
    """Decisive verdicts as a boolean stream in request order — CUSUM/SPRT food."""
    return [v["winner"] == "b" for v in decisive(verdicts)]


def measured_position_bias(verdicts: list[dict]) -> float:
    """P(winner was shown first | decisive) - 0.5.

    Zero for an unbiased judge; with randomized presentation this measures the
    judge's positional preference directly, without confounding from config
    quality (each config is first equally often).
    """
    d = decisive(verdicts)
    if not d:
        raise ValueError("no decisive verdicts")
    first_wins = sum(v["winner"] == v["first_shown"] for v in d) / len(d)
    return first_wins - 0.5
