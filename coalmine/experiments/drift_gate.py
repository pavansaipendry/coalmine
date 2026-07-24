"""Phase 5 experiment: the drift gate — "traffic changed, not the model."

The challenger has a stable, topic-dependent weakness (ε = 15% on "coding",
from t = 0). Neither config ever changes. At request 15,000 the traffic mix
shifts toward coding (25% → 70%) — and the aggregate win rate drops, because
the challenger's weak topic is suddenly overweighted. An ungated CUSUM raises
a QUALITY alarm and blames the challenger; the drift monitor sees the mix
shift within a few hundred requests, fires first, resets the in-flight test
(its iid assumption just died), and re-baselines. Per-topic win rates confirm
the correct attribution: within every topic, nothing changed.

A homogeneous control run (no topic weakness) takes the same traffic shift:
drift alarm fires (the mix did change), and no quality alarm ever does — the
gate distinguishes the two cases rather than suppressing alarms.

Run:  python -m coalmine.experiments.drift_gate
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
from coalmine.fleet.drift import InputDriftMonitor, calibrate_psi_threshold
from coalmine.judging.judges import NoisyOracleJudge
from coalmine.judging.pipeline import JudgePipeline
from coalmine.sim.runner import calibrate_cusum_threshold
from coalmine.sim.verdicts import gen_wins
from coalmine.stats.cusum import CUSUM
from coalmine.traffic.dataset import DEFAULT_TOPICS, synthetic_queries
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario, run_simulation

N_REQUESTS = 30_000
SHIFT_AT = 15_000
EPSILON = 0.15
WEAK_TOPIC = "coding"
SHADOW_RATE = 0.25
SHIFTED_WEIGHTS = {"coding": 0.7, "science": 0.1, "history": 0.1, "health": 0.1}
JUDGE = dict(accuracy=0.85, tie_rate=0.10, position_bias=0.15)
REBASELINE_VERDICTS = 600


async def _simulate(store, run_id, seed, heterogeneous: bool):
    queries = synthetic_queries(400)
    pool = ResponsePool.build_synthetic(queries)
    scenario = Scenario(
        run_id=run_id,
        seed=seed,
        n_requests=N_REQUESTS,
        shadow_rate=SHADOW_RATE,
        epsilon=EPSILON if heterogeneous else 0.0,
        changepoint=0,
        regressed_topic=WEAK_TOPIC if heterogeneous else None,
        topic_weights_after=SHIFTED_WEIGHTS,
        topic_shift_at=SHIFT_AT,
    )
    await run_simulation(scenario, queries, pool, store)
    verdicts = await JudgePipeline(NoisyOracleJudge(**JUDGE), seed=seed).judge_run(store, run_id)
    topic_of = {
        e.payload["request_index"]: e.payload["topic"]
        for e in store.read(run_id)
        if e.type == REQUEST_RECEIVED
    }
    return verdicts, topic_of


def _drift_alarm_index(topic_of: dict, threshold: float) -> int | None:
    monitor = InputDriftMonitor(
        sorted(DEFAULT_TOPICS), window=1_000, stride=100, threshold=threshold
    )
    for i in range(N_REQUESTS):
        if monitor.observe(topic_of[i]):
            return i
    return None


def _cusum_walk(decisive, h, p0=0.5, start=0, gate_at=None):
    """Walk a CUSUM over (request_index, win) pairs; optional drift gate:
    at the gate, reset, re-estimate the baseline from the next
    REBASELINE_VERDICTS verdicts, then resume with the new p0."""
    detector = CUSUM(p0=p0, p1=p0 - 0.05, h=h)
    xs, ys, alarm = [], [], None
    rebaseline: list[bool] = []
    state = "run"
    for index, win in decisive:
        if index < start:
            continue
        if gate_at is not None and state == "run" and index >= gate_at:
            state = "estimate"
            detector.reset()
        if state == "estimate":
            rebaseline.append(win)
            xs.append(index)
            ys.append(0.0)
            if len(rebaseline) >= REBASELINE_VERDICTS:
                p0_new = float(np.clip(np.mean(rebaseline), 0.10, 0.90))
                detector = CUSUM(p0=p0_new, p1=p0_new - 0.05, h=h)
                state = "run2"
            continue
        alarmed = detector.update(win)
        xs.append(index)
        ys.append(min(detector.stat / h, 1.0))
        if alarmed and alarm is None:
            alarm = index
            break
    return xs, ys, alarm


async def run(seed: int) -> dict:
    tmp = Path(tempfile.mkdtemp())
    store = SQLiteEventStore(tmp / "drift_gate.db")

    print("  heterogeneous run (challenger weak on coding) ...")
    verdicts, topic_of = await _simulate(store, "drift-hetero", seed, True)
    print("  homogeneous control ...")
    verdicts_ctl, topic_ctl = await _simulate(store, "drift-control", seed + 1, False)
    store.close()

    threshold = calibrate_psi_threshold(
        np.full(4, 0.25), 1_000, 100, N_REQUESTS - 1_000, 0.05, trials=400, seed=seed
    )
    rng = np.random.default_rng(seed + 2)
    horizon = int(N_REQUESTS * SHADOW_RATE * 0.92)
    h = calibrate_cusum_threshold(gen_wins(rng, 4_000, horizon, 0.5), 0.5, 0.45, 0.05)

    decisive = [(v["request_index"], v["winner"] == "b") for v in verdicts if v["winner"] != "tie"]
    decisive_ctl = [
        (v["request_index"], v["winner"] == "b") for v in verdicts_ctl if v["winner"] != "tie"
    ]

    drift_at = _drift_alarm_index(topic_of, threshold)
    drift_at_ctl = _drift_alarm_index(topic_ctl, threshold)
    ungated_x, ungated_y, ungated_alarm = _cusum_walk(decisive, h)
    gated_x, gated_y, gated_alarm = _cusum_walk(decisive, h, gate_at=drift_at)
    _, _, ctl_alarm = _cusum_walk(decisive_ctl, h)

    def topic_win_rate(pairs, topics, topic, lo, hi):
        wins = [w for i, w in pairs if topics[i] == topic and lo <= i < hi]
        return float(np.mean(wins)) if wins else None

    per_topic = {
        t: {
            "pre": topic_win_rate(decisive, topic_of, t, 0, SHIFT_AT),
            "post": topic_win_rate(decisive, topic_of, t, SHIFT_AT, N_REQUESTS),
        }
        for t in sorted(DEFAULT_TOPICS)
    }

    results = {
        "config": {
            "n_requests": N_REQUESTS,
            "shift_at": SHIFT_AT,
            "epsilon": EPSILON,
            "weak_topic": WEAK_TOPIC,
            "shifted_weights": SHIFTED_WEIGHTS,
            "judge": JUDGE,
            "seed": seed,
        },
        "drift_alarm_at": drift_at,
        "drift_detection_delay": (drift_at - SHIFT_AT) if drift_at else None,
        "ungated_quality_alarm_at": ungated_alarm,
        "gated_quality_alarm_at": gated_alarm,
        "control_drift_alarm_at": drift_at_ctl,
        "control_quality_alarm_at": ctl_alarm,
        "per_topic_win_rates": per_topic,
        "paths": {"ungated": [ungated_x, ungated_y], "gated": [gated_x, gated_y]},
    }
    print(
        f"  drift alarm @ {drift_at:,} (+{results['drift_detection_delay']} after shift) · "
        f"ungated quality alarm @ {ungated_alarm:,} · gated: {gated_alarm}"
    )
    print(
        f"  control: drift alarm @ {drift_at_ctl:,}, quality alarm: {ctl_alarm} · "
        f"coding win rate pre {per_topic[WEAK_TOPIC]['pre']:.3f} vs post "
        f"{per_topic[WEAK_TOPIC]['post']:.3f}"
    )
    return results


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    cfg = results["config"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.8, 6.4), sharex=True, height_ratios=[1, 1.4]
    )
    for ax in (ax1, ax2):
        style.style_axes(ax)

    share = cfg["shifted_weights"][cfg["weak_topic"]]
    ax1.plot(
        [0, cfg["shift_at"], cfg["shift_at"], cfg["n_requests"]],
        [0.25, 0.25, share, share],
        color=style.SERIES_1,
        linewidth=1.8,
    )
    drift_at = results["drift_alarm_at"]
    ax1.plot([drift_at], [share], marker="o", markersize=7, color=style.SERIES_1)
    ax1.annotate(
        f"drift alarm @ {drift_at:,}\n(+{results['drift_detection_delay']} requests)",
        (drift_at, share),
        textcoords="offset points",
        xytext=(10, -26),
        color=style.INK,
        fontsize=9,
        fontweight="bold",
    )
    ax1.set_ylabel(f'"{cfg["weak_topic"]}" traffic share')
    ax1.set_ylim(0, 0.85)
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style.titles(
        ax1,
        "Traffic changed, not the model — and the system says so",
        f"Challenger has a stable weakness on {cfg['weak_topic']!r} (ε = "
        f"{cfg['epsilon']:.0%} since t = 0) · at request {cfg['shift_at']:,} the mix "
        "shifts toward it",
    )

    ux, uy = results["paths"]["ungated"]
    gx, gy = results["paths"]["gated"]
    ax2.plot(ux, uy, color=style.SERIES_1, linewidth=1.5, label="Ungated CUSUM (blames the model)")
    ax2.plot(gx, gy, color=style.SERIES_2, linewidth=1.5, label="Drift-gated (reset + re-baseline)")
    ax2.axhline(1.0, color=style.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax2.axvline(cfg["shift_at"], color=style.GRID, linewidth=1.0)
    ungated_alarm = results["ungated_quality_alarm_at"]
    ax2.plot([ungated_alarm], [1.0], marker="o", markersize=7, color=style.SERIES_1)
    ax2.annotate(
        f"false quality alarm @ {ungated_alarm:,}",
        (ungated_alarm, 1.0),
        textcoords="offset points",
        xytext=(-8, -18),
        ha="right",
        color=style.INK,
        fontsize=9,
        fontweight="bold",
    )
    coding = results["per_topic_win_rates"][cfg["weak_topic"]]
    ax2.set_xlabel("Request index")
    ax2.set_ylabel("CUSUM statistic / threshold")
    ax2.set_ylim(0, 1.12)
    style.titles(
        ax2,
        "The gate turns a false blame into a correct attribution",
        f"Within-topic quality never moved ({cfg['weak_topic']}: "
        f"{coding['pre']:.1%} pre vs {coding['post']:.1%} post)\n"
        "Homogeneous control: drift alarm fires, quality alarm never does",
    )
    ax2.legend(frameon=False, loc="upper left", labelcolor=style.INK_2, fontsize=9)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(out_dir / "drift_gate.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = asyncio.run(run(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "drift_gate.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
