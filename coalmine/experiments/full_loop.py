"""Phase 3 headline: the full loop catches an injected regression end to end.

First composition of all three phases: simulated traffic flows through the
shadow router (Phase 2), a noisy position-randomized judge turns sampled
response pairs into verdicts (Phase 3), and the calibrated CUSUM watches the
decisive-verdict stream (Phase 1). A 15% ε-mixture regression is injected at
request 10,000; a paired null run (ε=0) shows the detector staying quiet on
healthy traffic. Every stage reads from and writes to the same event log, so
the whole story — traffic, verdicts, detection — is replayable from Postgres/
SQLite alone.

Run:  python -m coalmine.experiments.full_loop
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.core.store import SQLiteEventStore
from coalmine.experiments import style
from coalmine.judging.judges import NoisyOracleJudge
from coalmine.judging.metrics import challenger_win_rate, measured_position_bias
from coalmine.judging.pipeline import JudgePipeline
from coalmine.sim.runner import calibrate_cusum_threshold
from coalmine.sim.verdicts import gen_wins
from coalmine.stats.cusum import CUSUM
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario, run_simulation

N_REQUESTS = 30_000
CHANGEPOINT = 10_000
EPSILON = 0.15
SHADOW_RATE = 0.25
P0, P1_DESIGN = 0.50, 0.45
FALSE_ALARM_BUDGET = 0.05
CALIB_TRIALS = 4_000
JUDGE = dict(accuracy=0.85, tie_rate=0.10, position_bias=0.15)


def _cusum_path(verdicts: list[dict], h: float) -> tuple[list[int], list[float], int | None]:
    """Walk the scalar CUSUM over decisive verdicts in request order; stop at
    the alarm. Returns (request indices, statistic path, alarm request)."""
    detector = CUSUM(p0=P0, p1=P1_DESIGN, h=h)
    xs, ys = [], []
    for v in verdicts:
        if v["winner"] == "tie":
            continue
        alarmed = detector.update(v["winner"] == "b")
        xs.append(v["request_index"])
        ys.append(min(detector.stat, h))
        if alarmed:
            return xs, ys, v["request_index"]
    return xs, ys, None


async def run(seed: int) -> dict:
    queries = synthetic_queries(400)
    pool = ResponsePool.build_synthetic(queries)
    store = SQLiteEventStore(Path(tempfile.mkdtemp()) / "full_loop.db")
    judge = NoisyOracleJudge(**JUDGE)

    runs = {}
    for run_id, eps in (("full-loop-regressed", EPSILON), ("full-loop-null", 0.0)):
        scenario = Scenario(
            run_id=run_id,
            seed=seed,
            n_requests=N_REQUESTS,
            shadow_rate=SHADOW_RATE,
            epsilon=eps,
            changepoint=CHANGEPOINT if eps else None,
        )
        await run_simulation(scenario, queries, pool, store)
        pipeline = JudgePipeline(judge, seed=seed)
        runs[run_id] = await pipeline.judge_run(store, run_id)
        print(f"  {run_id}: {len(runs[run_id])} verdicts judged")

    null_verdicts = runs["full-loop-null"]
    reg_verdicts = runs["full-loop-regressed"]
    n_decisive_null = sum(v["winner"] != "tie" for v in null_verdicts)

    print(f"  calibrating CUSUM threshold over a {n_decisive_null}-verdict horizon ...")
    rng = np.random.default_rng(seed + 1)
    calib = gen_wins(rng, CALIB_TRIALS, n_decisive_null, P0)
    h = calibrate_cusum_threshold(calib, P0, P1_DESIGN, FALSE_ALARM_BUDGET)

    reg_x, reg_y, alarm_at = _cusum_path(reg_verdicts, h)
    null_x, null_y, null_alarm = _cusum_path(null_verdicts, h)

    pre = [v for v in reg_verdicts if v["request_index"] < CHANGEPOINT]
    post = [v for v in reg_verdicts if v["request_index"] >= CHANGEPOINT]
    results = {
        "config": {
            "n_requests": N_REQUESTS,
            "changepoint": CHANGEPOINT,
            "epsilon": EPSILON,
            "shadow_rate": SHADOW_RATE,
            "judge": JUDGE,
            "cusum": {"p0": P0, "p1": P1_DESIGN, "h": h},
            "false_alarm_budget": FALSE_ALARM_BUDGET,
            "seed": seed,
        },
        "win_rate_pre_changepoint": challenger_win_rate(pre),
        "win_rate_post_changepoint": challenger_win_rate(post),
        "measured_position_bias": measured_position_bias(reg_verdicts),
        "alarm_request_index": alarm_at,
        "detection_latency_requests": (alarm_at - CHANGEPOINT) if alarm_at else None,
        "null_run_alarmed": null_alarm is not None,
        "paths": {
            "regressed": [reg_x, reg_y],
            "null": [null_x, null_y],
        },
    }
    store.close()
    print(
        f"  challenger win rate: {results['win_rate_pre_changepoint']:.3f} pre-changepoint, "
        f"{results['win_rate_post_changepoint']:.3f} post"
    )
    if alarm_at:
        print(
            f"  ALARM at request {alarm_at:,} — "
            f"{results['detection_latency_requests']:,} requests after the injection"
        )
    else:
        print("  no alarm fired on the regressed run (missed)")
    print(f"  null run alarmed: {results['null_run_alarmed']}")
    return results


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    style.style_axes(ax)
    cfg = results["config"]
    h = cfg["cusum"]["h"]

    null_x, null_y = results["paths"]["null"]
    reg_x, reg_y = results["paths"]["regressed"]
    ax.plot(reg_x, reg_y, color=style.SERIES_1, linewidth=1.6, label="Regressed run (ε = 15%)")
    ax.plot(null_x, null_y, color=style.SERIES_2, linewidth=1.4, label="Null run (ε = 0)")
    ax.axhline(h, color=style.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(
        N_REQUESTS - 300, h + 0.12, "alarm threshold",
        color=style.MUTED, fontsize=8.5, va="bottom", ha="right",
    )
    ax.axvline(cfg["changepoint"], color=style.GRID, linewidth=1.0)
    ax.text(
        cfg["changepoint"] + 300, h * 0.55, "regression\ninjected",
        color=style.INK_2, fontsize=8.5,
    )
    alarm = results["alarm_request_index"]
    if alarm:
        ax.plot([alarm], [h], marker="o", markersize=7, color=style.SERIES_1)
        ax.annotate(
            f"alarm @ {alarm:,}",
            (alarm, h),
            textcoords="offset points",
            xytext=(10, -14),
            color=style.INK,
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("Request index")
    ax.set_ylabel("CUSUM statistic")
    style.titles(
        ax,
        "Traffic → judge → sequential detection, end to end",
        f"Noisy judge (accuracy {cfg['judge']['accuracy']:.0%}, ties "
        f"{cfg['judge']['tie_rate']:.0%}, position bias {cfg['judge']['position_bias']:.0%}, "
        f"randomized) · {cfg['shadow_rate']:.0%} shadow sampling\n"
        f"Win rate {results['win_rate_pre_changepoint']:.1%} → "
        f"{results['win_rate_post_changepoint']:.1%} after injection · detected "
        f"{results['detection_latency_requests']:,} requests later · null run stayed quiet",
    )
    ax.legend(frameon=False, loc="upper left", labelcolor=style.INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "full_loop.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = asyncio.run(run(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "full_loop.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
