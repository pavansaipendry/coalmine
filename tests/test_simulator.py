from coalmine.core.events import REQUEST_RECEIVED, RUN_FINISHED, RUN_STARTED
from coalmine.core.replay import content_fingerprint, measured_epsilon
from coalmine.core.store import SQLiteEventStore
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario, run_simulation


def _fixtures():
    queries = synthetic_queries(40)
    return queries, ResponsePool.build_synthetic(queries)


async def test_same_seed_same_content_different_seed_differs(tmp_path):
    queries, pool = _fixtures()
    store = SQLiteEventStore(tmp_path / "det.db")
    fps = {}
    for run_id, seed in (("run-a", 5), ("run-b", 5), ("run-c", 6)):
        scenario = Scenario(run_id=run_id, seed=seed, n_requests=300, shadow_rate=0.3)
        await run_simulation(scenario, queries, pool, store)
        fps[run_id] = content_fingerprint(store.read(run_id))
    assert fps["run-a"] == fps["run-b"], "same seed must reproduce identical content"
    assert fps["run-a"] != fps["run-c"], "different seed must not"
    store.close()


async def test_injected_epsilon_recoverable_from_log(tmp_path):
    queries, pool = _fixtures()
    store = SQLiteEventStore(tmp_path / "eps.db")
    scenario = Scenario(
        run_id="eps",
        seed=11,
        n_requests=2000,
        shadow_rate=1.0,
        epsilon=0.2,
        changepoint=1000,
    )
    result = await run_simulation(scenario, queries, pool, store)
    events = store.read("eps")
    assert measured_epsilon(events, "challenger", 0, 1000) == 0.0
    post = measured_epsilon(events, "challenger", 1000, 2000)
    assert abs(post - 0.2) < 0.04
    assert result.counts["shadow_response"] == 2000
    store.close()


async def test_topic_shift_is_exact_with_onehot_weights(tmp_path):
    queries, pool = _fixtures()
    store = SQLiteEventStore(tmp_path / "topics.db")
    scenario = Scenario(
        run_id="shift",
        seed=3,
        n_requests=400,
        shadow_rate=0.0,
        topic_weights={"science": 1.0},
        topic_weights_after={"coding": 1.0},
        topic_shift_at=200,
    )
    await run_simulation(scenario, queries, pool, store)
    topics = {
        e.payload["request_index"]: e.payload["topic"]
        for e in store.read("shift")
        if e.type == REQUEST_RECEIVED
    }
    assert all(topics[i] == "science" for i in range(200))
    assert all(topics[i] == "coding" for i in range(200, 400))
    store.close()


async def test_run_lifecycle_events_and_counts(tmp_path):
    queries, pool = _fixtures()
    store = SQLiteEventStore(tmp_path / "life.db")
    result = await run_simulation(
        Scenario(run_id="life", seed=1, n_requests=100, shadow_rate=0.5), queries, pool, store
    )
    events = store.read("life")
    types = [e.type for e in events]
    assert types.count(RUN_STARTED) == 1
    assert types.count(RUN_FINISHED) == 1
    assert types.count(REQUEST_RECEIVED) == 100
    assert result.counts["champion_response"] == 100
    assert len(result.user_latency_ms) == 100
    finished = next(e for e in events if e.type == RUN_FINISHED)
    assert finished.payload["counts"]["champion_response"] == 100
    store.close()


async def test_topic_confined_regression(tmp_path):
    queries, pool = _fixtures()
    store = SQLiteEventStore(tmp_path / "topic_reg.db")
    scenario = Scenario(
        run_id="topic-reg",
        seed=9,
        n_requests=600,
        shadow_rate=1.0,
        epsilon=1.0,  # deterministic: every regressed request serves bad
        changepoint=300,
        regressed_topic="coding",
    )
    await run_simulation(scenario, queries, pool, store)
    events = store.read("topic-reg")
    topic_of = {
        e.payload["request_index"]: e.payload["topic"]
        for e in events
        if e.type == REQUEST_RECEIVED
    }
    for e in events:
        if e.type != "shadow_response":
            continue
        idx = e.payload["request_index"]
        expect_bad = topic_of[idx] == "coding" and idx >= 300
        assert (e.payload["source_pool"] == "bad") == expect_bad, f"request {idx}"
    store.close()


async def test_paced_mode_runs_concurrently(tmp_path):
    queries, pool = _fixtures()
    store = SQLiteEventStore(tmp_path / "paced.db")
    result = await run_simulation(
        Scenario(run_id="paced", seed=2, n_requests=40, rate_rps=400.0, shadow_rate=0.2),
        queries,
        pool,
        store,
    )
    assert result.counts["champion_response"] == 40
    assert len(result.user_latency_ms) == 40
    store.close()
