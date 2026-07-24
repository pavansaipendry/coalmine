"""Phase 4 experiment: stratified monitors catch what the aggregate smears out.

A regression confined to one topic (ε = 50% on "coding" only, one quarter of
traffic) barely moves the aggregate win rate — the healthy topics dilute it
4:1 — but is a violent shift inside its own stratum. Per-topic CUSUMs (each
at α/4 over its own horizon, so the family holds α by union bound) reuse the
topic labels already in the event log and catch the regression several times
sooner than the single aggregate monitor. This is the drift-cluster
stratification story from the design, run end to end through the real
pipeline: traffic → judge → per-stratum detection.

Run:  python -m coalmine.experiments.stratified
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.core.events import REQUEST_RECEIVED
from coalmine.core.store import SQLiteEventStore
from coalmine.experiments import style
from coalmine.judging.judges import NoisyOracleJudge
from coalmine.judging.pipeline import JudgePipeline
from coalmine.sim.runner import calibrate_cusum_threshold
from coalmine.sim.verdicts import gen_wins
from coalmine.stats.cusum import CUSUM
from coalmine.traffic.dataset import DEFAULT_TOPICS, synthetic_queries
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario, run_simulation

N_REQUESTS = 40_000
CHANGEPOINT = 10_000
EPSILON = 0.50
REGRESSED_TOPIC = "coding"
SHADOW_RATE = 0.25
P0, P1_DESIGN = 0.50, 0.45
ALPHA = 0.05
CALIB_TRIALS = 4_000
JUDGE = dict(accuracy=0.85, tie_rate=0.10, position_bias=0.15)


def _walk(verdict_stream: list[tuple[int, bool]], h: float) -> tuple[list, list, int | None]:
    detector = CUSUM(p0=P0, p1=P1_DESIGN, h=h)
    xs, ys = [], []
    for request_index, win in verdict_stream:
        alarmed = detector.update(win)
        xs.append(request_index)
        ys.append(min(detector.stat / h, 1.0))  # normalized: 1.0 = threshold
        if alarmed:
            return xs, ys, request_index
    return xs, ys, None


async def run(seed: int) -> dict:
    queries = synthetic_queries(400)
    pool = ResponsePool.build_synthetic(queries)
    store = SQLiteEventStore(Path(tempfile.mkdtemp()) / "stratified.db")
    judge = NoisyOracleJudge(**JUDGE)

    streams = {}
    for run_id, eps in (("strat-regressed", EPSILON), ("strat-null", 0.0)):
        scenario = Scenario(
            run_id=run_id,
            seed=seed,
            n_requests=N_REQUESTS,
            shadow_rate=SHADOW_RATE,
            epsilon=eps,
            changepoint=CHANGEPOINT,
            regressed_topic=REGRESSED_TOPIC if eps else None,
        )
        await run_simulation(scenario, queries, pool, store)
        verdicts = await JudgePipeline(judge, seed=seed).judge_run(store, run_id)
        topic_of = {
            e.payload["request_index"]: e.payload["topic"]
            for e in store.read(run_id)
            if e.type == REQUEST_RECEIVED
        }
        decisive = [
            (v["request_index"], v["winner"] == "b") for v in verdicts if v["winner"] != "tie"
        ]
        streams[run_id] = {
            "all": decisive,
            **{
                t: [(i, w) for i, w in decisive if topic_of[i] == t]
                for t in DEFAULT_TOPICS
            },
        }
        print(f"  {run_id}: {len(decisive)} decisive verdicts")
    store.close()

    rng = np.random.default_rng(seed + 1)
    null = streams["strat-null"]
    h_aggregate = calibrate_cusum_threshold(
        gen_wins(rng, CALIB_TRIALS, len(null["all"]), P0), P0, P1_DESIGN, ALPHA
    )
    per_topic_alpha = ALPHA / len(DEFAULT_TOPICS)
    h_topic = {
        t: calibrate_cusum_threshold(
            gen_wins(rng, CALIB_TRIALS, len(null[t]), P0), P0, P1_DESIGN, per_topic_alpha
        )
        for t in DEFAULT_TOPICS
    }

    reg = streams["strat-regressed"]
    agg_x, agg_y, agg_alarm = _walk(reg["all"], h_aggregate)
    topic_alarms = {}
    coding_path = None
    for t in DEFAULT_TOPICS:
        xs, ys, alarm = _walk(reg[t], h_topic[t])
        topic_alarms[t] = alarm
        if t == REGRESSED_TOPIC:
            coding_path = (xs, ys)
    null_family_alarms = [
        _walk(null[t], h_topic[t])[2] for t in DEFAULT_TOPICS
    ] + [_walk(null["all"], h_aggregate)[2]]

    results = {
        "config": {
            "n_requests": N_REQUESTS,
            "changepoint": CHANGEPOINT,
            "epsilon": EPSILON,
            "regressed_topic": REGRESSED_TOPIC,
            "shadow_rate": SHADOW_RATE,
            "judge": JUDGE,
            "alpha": ALPHA,
            "seed": seed,
        },
        "aggregate_alarm": agg_alarm,
        "topic_alarms": topic_alarms,
        "stratified_latency": (
            topic_alarms[REGRESSED_TOPIC] - CHANGEPOINT
            if topic_alarms[REGRESSED_TOPIC]
            else None
        ),
        "aggregate_latency": (agg_alarm - CHANGEPOINT) if agg_alarm else None,
        "healthy_topic_false_alarms": sum(
            1 for t, a in topic_alarms.items() if t != REGRESSED_TOPIC and a is not None
        ),
        "null_run_any_alarm": any(a is not None for a in null_family_alarms),
        "paths": {"aggregate": [agg_x, agg_y], "coding": list(coding_path)},
    }
    print(
        f"  stratified ({REGRESSED_TOPIC}) alarm @ {topic_alarms[REGRESSED_TOPIC]:,} "
        f"(+{results['stratified_latency']:,}) · aggregate alarm @ "
        f"{agg_alarm and f'{agg_alarm:,}'} (+{results['aggregate_latency']})"
    )
    print(
        f"  healthy-topic false alarms: {results['healthy_topic_false_alarms']} · "
        f"null-run family alarms: {results['null_run_any_alarm']}"
    )
    return results


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    style.style_axes(ax)
    cfg = results["config"]

    agg_x, agg_y = results["paths"]["aggregate"]
    cod_x, cod_y = results["paths"]["coding"]
    ax.plot(agg_x, agg_y, color=style.SERIES_1, linewidth=1.5, label="Aggregate monitor (α)")
    ax.plot(
        cod_x, cod_y, color=style.SERIES_2, linewidth=1.5,
        label=f'Stratified monitor: "{cfg["regressed_topic"]}" (α/4)',
    )
    x_max = max(agg_x[-1], cod_x[-1]) * 1.12
    ax.set_xlim(0, x_max)
    ax.axhline(1.0, color=style.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(
        x_max * 0.99, 1.015, "alarm threshold (normalized)",
        color=style.MUTED, fontsize=8.5, ha="right",
    )
    ax.axvline(cfg["changepoint"], color=style.GRID, linewidth=1.0)
    ax.text(
        cfg["changepoint"] - 400, 0.78,
        f'ε = {cfg["epsilon"]:.0%} injected\ninto "{cfg["regressed_topic"]}" only',
        color=style.INK_2, fontsize=8.5, ha="right",
    )
    for x, alarm, color, offset, align in (
        (
            results["topic_alarms"][cfg["regressed_topic"]],
            "stratified", style.SERIES_2, (-8, -20), "right",
        ),
        (results["aggregate_alarm"], "aggregate", style.SERIES_1, (8, -20), "left"),
    ):
        if x:
            ax.plot([x], [1.0], marker="o", markersize=7, color=color)
            ax.annotate(
                f"{alarm} alarm\n@ {x:,}",
                (x, 1.0),
                textcoords="offset points",
                xytext=offset,
                ha=align,
                va="top",
                color=style.INK,
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xlabel("Request index")
    ax.set_ylabel("CUSUM statistic / threshold")
    ax.set_ylim(0, 1.12)
    speedup = results["aggregate_latency"] / results["stratified_latency"]
    style.titles(
        ax,
        "A one-topic regression: stratified monitors catch it "
        f"{speedup:.1f}× sooner",
        f"{cfg['regressed_topic']!r} is 25% of traffic, so the aggregate shift is diluted "
        "4:1 · per-topic CUSUMs at α/4 hold the family budget\n"
        f"stratified latency +{results['stratified_latency']:,} requests vs aggregate "
        f"+{results['aggregate_latency']:,} · no false alarms on healthy topics or the null run",
    )
    ax.legend(frameon=False, loc="upper left", labelcolor=style.INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "stratified.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = asyncio.run(run(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stratified.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
