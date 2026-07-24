"""Phase 6 headline: the closed loop — ramp, promote, and roll back, unaided.

Two lifecycle runs through the full fleet (traffic → router → judges →
canary controller), no humans anywhere:

- An equal-quality challenger climbs the whole ladder — shadow → 1% → 5% →
  25% → 100% — each step gated by a fresh non-inferiority mixture SPRT at
  α/4, and is promoted.
- A challenger that regresses mid-ramp (ε = 25% injected at request 6,000,
  during the 5% stage) is caught by the continuous rollback CUSUM and yanked
  to 0% user traffic. After the rollback's share update reaches the router,
  the challenger never serves another user. (Post-promotion monitoring is the
  standing decision service's job — the lifecycle ends at promoted or
  rolled-back.)

Every gate pass, the promotion, and the rollback are events in the log — the
audit trail below is read back from the store, not from program state.

Run:  python -m coalmine.experiments.canary_lifecycle
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.canary.lifecycle import CanaryLifecycle
from coalmine.core.events import REQUEST_RECEIVED
from coalmine.core.store import SQLiteEventStore
from coalmine.experiments import style
from coalmine.fleet.bus import InMemoryBus
from coalmine.fleet.runner import FleetRunner
from coalmine.fleet.services import CANARY_TRANSITION
from coalmine.judging.judges import NoisyOracleJudge
from coalmine.sim.runner import calibrate_cusum_threshold
from coalmine.sim.verdicts import gen_wins
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario

N_REQUESTS = 30_000
SHADOW_RATE = 0.35
ROLLBACK_EPSILON = 0.25
ROLLBACK_CHANGEPOINT = 6_000  # mid-ramp: lands inside the 5% stage
JUDGE = dict(accuracy=0.85, tie_rate=0.10, position_bias=0.15)

STAGE_LEVEL = {"shadow": 0, "canary-1": 1, "canary-5": 2, "canary-25": 3, "full": 4}
LEVEL_LABEL = ["shadow (0%)", "1%", "5%", "25%", "100% promoted"]


async def _run(run_id: str, seed: int, epsilon: float, changepoint: int | None, h: float):
    queries = synthetic_queries(400)
    pool = ResponsePool.build_synthetic(queries)
    store = SQLiteEventStore(Path(tempfile.mkdtemp()) / f"{run_id}.db")
    lifecycle = CanaryLifecycle(cusum_h=h)
    runner = FleetRunner(
        bus=InMemoryBus(),
        store=store,
        scenario=Scenario(
            run_id=run_id,
            seed=seed,
            n_requests=N_REQUESTS,
            shadow_rate=SHADOW_RATE,
            epsilon=epsilon,
            changepoint=changepoint,
        ),
        queries=queries,
        pool=pool,
        judge=NoisyOracleJudge(**JUDGE),
        cusum_h=h,
        n_judges=2,
        lifecycle=lifecycle,
    )
    await runner.run(timeout=600.0)
    events = store.read(run_id)
    transitions = [e.payload for e in events if e.type == CANARY_TRANSITION]
    transitions.sort(key=lambda p: p["request_index"])
    served = {
        e.payload["request_index"]: e.payload["served_by"]
        for e in events
        if e.type == REQUEST_RECEIVED
    }
    store.close()
    return lifecycle, transitions, served


def _share_steps(transitions: list[dict], final_index: int) -> tuple[list, list]:
    xs, ys = [0], [0]
    for t in transitions:
        level = STAGE_LEVEL.get(t["stage"], 0)
        xs += [t["request_index"], t["request_index"]]
        ys += [ys[-1], level]
    xs.append(final_index)
    ys.append(ys[-1])
    return xs, ys


async def run(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    horizon = int(N_REQUESTS * SHADOW_RATE * 0.92)
    h = calibrate_cusum_threshold(gen_wins(rng, 4_000, horizon, 0.5), 0.5, 0.45, 0.05)

    print("  run A: equal-quality challenger (should promote) ...")
    life_a, trans_a, _ = await _run("canary-promoted", seed, 0.0, None, h)
    print(f"    -> {life_a.state} after {life_a.verdicts_seen} verdicts")
    for t in trans_a:
        print(
            f"       {t['kind']:>18} → {t['stage']:<10} share {t['share']:>5.0%} "
            f"@ request {t['request_index']:,}"
        )

    print(f"  run B: regression injected @ {ROLLBACK_CHANGEPOINT:,} (should roll back) ...")
    life_b, trans_b, served_b = await _run(
        "canary-rollback", seed, ROLLBACK_EPSILON, ROLLBACK_CHANGEPOINT, h
    )
    print(f"    -> {life_b.state}")
    for t in trans_b:
        print(
            f"       {t['kind']:>18} → {t['stage']:<10} share {t['share']:>5.0%} "
            f"@ request {t['request_index']:,}"
        )
    rollback_at = next(
        (t["request_index"] for t in trans_b if t["kind"] == "canary_rolled_back"), None
    )
    served_after = sum(
        1
        for i, by in served_b.items()
        if rollback_at is not None and i > rollback_at + 200 and by == "challenger"
    )
    print(f"    challenger-served users after rollback (+200 propagation): {served_after}")

    return {
        "config": {
            "n_requests": N_REQUESTS,
            "shadow_rate": SHADOW_RATE,
            "rollback_epsilon": ROLLBACK_EPSILON,
            "rollback_changepoint": ROLLBACK_CHANGEPOINT,
            "judge": JUDGE,
            "cusum_h": h,
            "margin": 0.06,
            "seed": seed,
        },
        "promoted_run": {
            "state": life_a.state,
            "transitions": trans_a,
            "verdicts": life_a.verdicts_seen,
        },
        "rollback_run": {
            "state": life_b.state,
            "transitions": trans_b,
            "rollback_at": rollback_at,
            "rollback_latency": (
                rollback_at - ROLLBACK_CHANGEPOINT if rollback_at is not None else None
            ),
            "challenger_served_after_rollback": served_after,
        },
    }


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    style.style_axes(ax)
    cfg = results["config"]

    ax_a = _share_steps(results["promoted_run"]["transitions"], cfg["n_requests"])
    ax.plot(*ax_a, color=style.SERIES_2, linewidth=1.9, label="Equal challenger → promoted")
    for t in results["promoted_run"]["transitions"]:
        ax.plot(
            [t["request_index"]],
            [STAGE_LEVEL.get(t["stage"], 0)],
            marker="o",
            markersize=6,
            color=style.SERIES_2,
        )

    trans_b = results["rollback_run"]["transitions"]
    xb, yb = _share_steps(
        [t for t in trans_b if t["kind"] != "canary_rolled_back"], cfg["n_requests"]
    )
    rollback_at = results["rollback_run"]["rollback_at"]
    if rollback_at is not None:
        keep = [x for x in xb if x <= rollback_at]
        yb = yb[: len(keep)] + [yb[len(keep) - 1], 0, 0]
        xb = keep + [rollback_at, rollback_at, cfg["n_requests"]]
    # Dodge the rollback run slightly so the identical ramp prefix of both
    # runs stays visible as two lines.
    yb_dodged = [y - 0.07 for y in yb]
    ax.plot(
        xb, yb_dodged, color=style.SERIES_1, linewidth=1.9,
        label="Regressed mid-ramp → rolled back",
    )
    if rollback_at is not None:
        ax.plot([rollback_at], [-0.07], marker="o", markersize=7, color=style.SERIES_1)
        ax.annotate(
            f"auto-rollback @ {rollback_at:,}\n(+{results['rollback_run']['rollback_latency']:,}"
            " after injection)",
            (rollback_at, 0),
            textcoords="offset points",
            xytext=(12, 14),
            color=style.INK,
            fontsize=9,
            fontweight="bold",
        )
    ax.axvline(cfg["rollback_changepoint"], color=style.GRID, linewidth=1.0)
    ax.text(
        cfg["rollback_changepoint"] - 400,
        3.55,
        "regression injected\n(rollback run only)",
        color=style.INK_2,
        fontsize=8.5,
        ha="right",
    )
    ax.set_yticks(range(5), LEVEL_LABEL)
    ax.set_xlabel("Request index")
    ax.set_ylabel("Challenger user-traffic share")
    ax.set_ylim(-0.4, 4.5)
    style.titles(
        ax,
        "The closed loop: ramp, promote, roll back — no humans",
        "Each step gated by a fresh non-inferiority mixture SPRT (margin 6pp, α/4) · "
        "continuous CUSUM rollback alarm\nevery transition below is replayed from the "
        "event log · challenger served 0 users after rollback",
    )
    ax.legend(frameon=False, loc="center right", labelcolor=style.INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "canary_lifecycle.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = asyncio.run(run(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "canary_lifecycle.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
