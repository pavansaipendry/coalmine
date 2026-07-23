import pytest

from coalmine.core.events import VERDICT_RECORDED
from coalmine.core.store import SQLiteEventStore
from coalmine.judging.judges import NoisyOracleJudge
from coalmine.judging.metrics import (
    challenger_win_rate,
    challenger_wins,
    measured_position_bias,
    verdicts_from_events,
)
from coalmine.judging.pipeline import JudgePipeline
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario, run_simulation


async def _simulated_store(tmp_path, run_id="judged", n=800, epsilon=0.0, changepoint=None):
    queries = synthetic_queries(40)
    pool = ResponsePool.build_synthetic(queries)
    store = SQLiteEventStore(tmp_path / "pipeline.db")
    scenario = Scenario(
        run_id=run_id,
        seed=31,
        n_requests=n,
        shadow_rate=1.0,
        epsilon=epsilon,
        changepoint=changepoint,
    )
    await run_simulation(scenario, queries, pool, store)
    return store


async def test_pipeline_appends_verdicts_to_same_run(tmp_path):
    store = await _simulated_store(tmp_path)
    pre_max = store.max_seq("judged")
    pipeline = JudgePipeline(NoisyOracleJudge(tie_rate=0.1), seed=7, sampling_rate=0.5)
    verdicts = await pipeline.judge_run(store, "judged")

    events = store.read("judged")
    stored = verdicts_from_events(events)
    assert stored == sorted(verdicts, key=lambda v: v["request_index"])
    assert 300 < len(verdicts) < 500  # ~50% of 800 sampled
    assert store.max_seq("judged") == pre_max + len(verdicts)
    verdict_events = [e for e in events if e.type == VERDICT_RECORDED]
    assert all(e.payload["judge_id"] == "noisy-oracle" for e in verdict_events)


async def test_pipeline_deterministic_per_judge_id_and_independent_across(tmp_path):
    store = await _simulated_store(tmp_path)
    judge = NoisyOracleJudge(tie_rate=0.1, position_bias=0.1)
    # Same judge_id: RNG streams are keyed by it, so a re-run reproduces
    # identical verdicts exactly.
    v1 = await JudgePipeline(judge, seed=7, judge_id="same").judge_run(store, "judged")
    v2 = await JudgePipeline(judge, seed=7, judge_id="same").judge_run(store, "judged")
    assert v1 == v2
    # Different judge_id: an ensemble member must see independent sampling and
    # randomization, not a copy of another member's stream.
    v3 = await JudgePipeline(judge, seed=7, judge_id="other").judge_run(store, "judged")
    strip = lambda vs: [{k: v for k, v in p.items() if k != "judge_id"} for p in vs]  # noqa: E731
    assert strip(v3) != strip(v1)


async def test_equal_quality_run_judges_to_parity(tmp_path):
    store = await _simulated_store(tmp_path, n=2000)
    pipeline = JudgePipeline(NoisyOracleJudge(position_bias=0.2), seed=11)
    verdicts = await pipeline.judge_run(store, "judged")
    # ε=0: both configs serve the good pool — win rate ~50% even with a biased
    # judge, because presentation order is randomized.
    assert abs(challenger_win_rate(verdicts) - 0.5) < 0.04
    # ...and the bias itself is visible in position space.
    assert measured_position_bias(verdicts) > 0.05


async def test_regressed_run_shifts_win_rate_down(tmp_path):
    store = await _simulated_store(tmp_path, n=2000, epsilon=0.3, changepoint=0)
    pipeline = JudgePipeline(NoisyOracleJudge(accuracy=0.85, tie_rate=0.1), seed=13)
    verdicts = await pipeline.judge_run(store, "judged")
    # (1-ε)*0.5 + ε*(1-a) = 0.7*0.5 + 0.3*0.15 = 0.395
    assert challenger_win_rate(verdicts) < 0.45
    wins = challenger_wins(verdicts)
    assert len(wins) == len([v for v in verdicts if v["winner"] != "tie"])


async def test_sampling_rate_bounds():
    with pytest.raises(ValueError):
        JudgePipeline(NoisyOracleJudge(), seed=1, sampling_rate=0.0)
