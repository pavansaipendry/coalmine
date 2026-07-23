"""Phase 2 experiment: injected ground truth is recoverable from the event log.

A 20,000-request run injects a 10% ε-mixture regression into the challenger
at request 10,000. Reading the event log back and computing the rolling
bad-pool fraction over shadow responses must reproduce the configured step —
this is the property that makes Phase 3's detection claims scorable: the log
carries exact ground truth for what the detectors should have found.

Also verifies content determinism end to end: two same-seed runs produce
identical order-free content fingerprints.

Run:  python -m coalmine.experiments.epsilon_verification
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from coalmine.core.replay import content_fingerprint, measured_epsilon, rolling_epsilon
from coalmine.core.store import SQLiteEventStore
from coalmine.experiments import style
from coalmine.traffic.dataset import synthetic_queries
from coalmine.traffic.pools import ResponsePool
from coalmine.traffic.simulator import Scenario, run_simulation

N_REQUESTS = 20_000
CHANGEPOINT = 10_000
EPSILON = 0.10
SHADOW_RATE = 0.25
ROLLING_WINDOW = 500  # shadow responses


async def run(seed: int) -> tuple[dict, list]:
    queries = synthetic_queries(400)
    pool = ResponsePool.build_synthetic(queries)
    store = SQLiteEventStore(Path(tempfile.mkdtemp()) / "epsilon.db")

    scenario = Scenario(
        run_id="eps-main",
        seed=seed,
        n_requests=N_REQUESTS,
        shadow_rate=SHADOW_RATE,
        epsilon=EPSILON,
        changepoint=CHANGEPOINT,
    )
    result = await run_simulation(scenario, queries, pool, store)
    events = store.read("eps-main")
    pre = measured_epsilon(events, "challenger", 0, CHANGEPOINT)
    post = measured_epsilon(events, "challenger", CHANGEPOINT, N_REQUESTS)
    print(f"  injected ε=0 before request {CHANGEPOINT:,}: measured {pre:.4f}")
    print(f"  injected ε={EPSILON} after request {CHANGEPOINT:,}: measured {post:.4f}")

    fingerprints = []
    for run_id in ("det-a", "det-b"):
        s2 = Scenario(run_id=run_id, seed=seed + 1, n_requests=2_000, shadow_rate=0.3)
        await run_simulation(s2, queries, pool, store)
        fingerprints.append(content_fingerprint(store.read(run_id)))
    deterministic = fingerprints[0] == fingerprints[1]
    print(f"  same-seed content fingerprints identical: {deterministic}")

    centers, rolled = rolling_epsilon(events, "challenger", ROLLING_WINDOW)
    store.close()
    results = {
        "config": {
            "n_requests": N_REQUESTS,
            "changepoint": CHANGEPOINT,
            "epsilon": EPSILON,
            "shadow_rate": SHADOW_RATE,
            "rolling_window": ROLLING_WINDOW,
            "seed": seed,
        },
        "measured_epsilon_pre": pre,
        "measured_epsilon_post": post,
        "shadow_responses": result.counts.get("shadow_response", 0),
        "content_deterministic": deterministic,
        "fingerprint": fingerprints[0],
    }
    return results, [centers.tolist(), rolled.tolist()]


def chart(results: dict, rolling: list, out_dir: Path) -> None:
    style.setup()
    centers, rolled = rolling
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    style.style_axes(ax)
    ax.plot(
        [0, CHANGEPOINT, CHANGEPOINT, N_REQUESTS],
        [0.0, 0.0, EPSILON, EPSILON],
        color=style.SERIES_1,
        linewidth=2.0,
        label="Injected (ground truth)",
    )
    ax.plot(centers, rolled, color=style.SERIES_2, linewidth=1.4, label="Recovered from log")
    ax.set_xlabel("Request index")
    ax.set_ylabel("Bad-pool fraction ε")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style.titles(
        ax,
        "The event log carries exact ground truth",
        f"Rolling mean over {results['config']['rolling_window']} shadow responses "
        f"({results['config']['shadow_rate']:.0%} sampling) · measured ε after changepoint: "
        f"{results['measured_epsilon_post']:.3f} vs {EPSILON} injected",
    )
    ax.legend(frameon=False, loc="upper left", labelcolor=style.INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "epsilon_verification.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results, rolling = asyncio.run(run(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "epsilon_verification.json").write_text(json.dumps(results, indent=2))
    chart(results, rolling, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
