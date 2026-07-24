"""The real-judge study, at zero cost: a local reward model vs human ground truth.

Measures the judge channel on real data with no API and no spend: the judge
is a local CPU reward model (OpenAssistant DeBERTa), the anchors are LMSYS's
MT-Bench human judgments — real model responses with human-consensus labels.
Deliverables: the judge's measured accuracy against human preference (the
channel parameter every simulation takes as input), broken out by label
confidence (unanimous vs majority pairs, with the human agreement ceiling for
context); verification of the reward judge's by-construction position
invariance; per-verdict latency; and the planning-calculator impact of the
measured channel versus the 0.85 assumption.

Run:  python -m coalmine.experiments.reward_judge_study   (~10-15 min on CPU)
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coalmine.experiments import style
from coalmine.judging.anchors import wilson_interval
from coalmine.judging.mtbench import consensus_anchors, load_mtbench_rows
from coalmine.judging.reward import RewardModelJudge
from coalmine.stats.channel import JudgeChannel
from coalmine.stats.wald import plan

ASSUMED_ACCURACY = 0.85
SAMPLING_RATE = 0.25
SWAP_CHECK = 25


async def run(seed: int, max_pairs: int | None = None) -> dict:
    del seed  # everything here is deterministic: real data, deterministic judge
    print("  loading MT-Bench human judgments ...")
    anchors = consensus_anchors(load_mtbench_rows())
    if max_pairs:
        anchors = anchors[:max_pairs]
    multi = [a for a in anchors if a.votes_a + a.votes_b >= 2]
    human_agreement = float(
        np.mean([max(a.votes_a, a.votes_b) / (a.votes_a + a.votes_b) for a in multi])
    )
    print(
        f"  {len(anchors)} consensus pairs ({len(multi)} with 2+ decisive votes; "
        f"human vote agreement {human_agreement:.1%})"
    )

    print("  loading reward model ...")
    judge = RewardModelJudge()
    print(f"  judging with {judge.judge_id} on CPU ...")

    correct = 0
    unanimous_correct = unanimous_total = 0
    latencies = []
    verdicts = []
    started = time.time()
    for i, anchor in enumerate(anchors):
        t0 = time.perf_counter()
        verdict = await judge.compare(
            anchor.question,
            {"text": anchor.text_a, "source_pool": None},
            {"text": anchor.text_b, "source_pool": None},
        )
        latencies.append(time.perf_counter() - t0)
        predicted = "a" if verdict == "first" else "b"
        verdicts.append(predicted)
        hit = predicted == anchor.label
        correct += hit
        if min(anchor.votes_a, anchor.votes_b) == 0 and anchor.votes_a + anchor.votes_b >= 2:
            unanimous_total += 1
            unanimous_correct += hit
        if (i + 1) % 100 == 0:
            print(
                f"    {i + 1}/{len(anchors)} · running accuracy "
                f"{correct / (i + 1):.3f} · {np.mean(latencies):.2f}s/verdict"
            )

    swap_consistent = 0
    for anchor, original in zip(anchors[:SWAP_CHECK], verdicts[:SWAP_CHECK]):
        swapped = await judge.compare(
            anchor.question,
            {"text": anchor.text_b, "source_pool": None},
            {"text": anchor.text_a, "source_pool": None},
        )
        swapped_predicted = "b" if swapped == "first" else "a"
        swap_consistent += swapped_predicted == original

    n = len(anchors)
    accuracy = correct / n
    low, high = wilson_interval(correct, n)
    channel_measured = JudgeChannel(accuracy=accuracy, tie_rate=0.0)
    channel_assumed = JudgeChannel(accuracy=ASSUMED_ACCURACY, tie_rate=0.10)
    plan_measured = plan(0.5, 0.55, 0.05, 0.10, channel_measured, SAMPLING_RATE)
    plan_assumed = plan(0.5, 0.55, 0.05, 0.10, channel_assumed, SAMPLING_RATE)

    results = {
        "config": {
            "judge": judge.judge_id,
            "n_pairs": n,
            "assumed_accuracy": ASSUMED_ACCURACY,
            "sampling_rate": SAMPLING_RATE,
            "cost_usd": 0.0,
        },
        "human_agreement_ceiling": human_agreement,
        "accuracy": accuracy,
        "wilson_low": low,
        "wilson_high": high,
        "unanimous_accuracy": unanimous_correct / unanimous_total,
        "unanimous_pairs": unanimous_total,
        "swap_consistency": swap_consistent / SWAP_CHECK,
        "seconds_per_verdict": float(np.mean(latencies)),
        "wall_minutes": (time.time() - started) / 60.0,
        "planning": {
            "assumed_expected_requests_h1": plan_assumed["expected_requests_under_h1"],
            "measured_expected_requests_h1": plan_measured["expected_requests_under_h1"],
        },
    }
    print(
        f"  accuracy vs human consensus: {accuracy:.3f} [{low:.3f}, {high:.3f}] · "
        f"unanimous pairs: {results['unanimous_accuracy']:.3f} · "
        f"human ceiling {human_agreement:.3f}"
    )
    print(
        f"  position invariance: {swap_consistent}/{SWAP_CHECK} identical under swap · "
        f"{results['seconds_per_verdict']:.2f}s/verdict · $0.00 spent"
    )
    print(
        f"  planning impact (promote 0.55 @ {SAMPLING_RATE:.0%} sampling): "
        f"{plan_assumed['expected_requests_under_h1']:,.0f} requests assumed vs "
        f"{plan_measured['expected_requests_under_h1']:,.0f} at the measured channel"
    )
    return results


def chart(results: dict, out_dir: Path) -> None:
    style.setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), width_ratios=[1.45, 1])
    for ax in (ax1, ax2):
        style.style_axes(ax)

    labels = ["Assumed", "Measured\nall pairs", "Measured\nunanimous", "Human\nagreement"]
    values = [
        results["config"]["assumed_accuracy"],
        results["accuracy"],
        results["unanimous_accuracy"],
        results["human_agreement_ceiling"],
    ]
    colors = [style.BAND_1, style.SERIES_1, style.SERIES_1, style.BASELINE]
    bars = ax1.bar(range(4), values, width=0.55, color=colors)
    ax1.errorbar(
        [1],
        [results["accuracy"]],
        yerr=[[results["accuracy"] - results["wilson_low"]],
              [results["wilson_high"] - results["accuracy"]]],
        fmt="none",
        ecolor=style.INK_2,
        capsize=4,
        linewidth=1.2,
    )
    for bar, v in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.1%}",
            ha="center", color=style.INK, fontsize=10, fontweight="bold",
        )
    ax1.set_xticks(range(4), labels)
    ax1.set_ylim(0.5, 1.03)
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax1.set_ylabel("Judge accuracy vs human consensus")
    cfg = results["config"]
    style.titles(
        ax1,
        "The judge channel, measured on human ground truth",
        f"{cfg['judge']} · local CPU · {cfg['n_pairs']} MT-Bench pairs · $0.00 spent\n"
        f"position-invariant by construction "
        f"({results['swap_consistency']:.0%} swap-consistent)",
    )

    plans = results["planning"]
    values2 = [plans["assumed_expected_requests_h1"], plans["measured_expected_requests_h1"]]
    bars = ax2.bar(range(2), values2, width=0.5, color=[style.BAND_1, style.SERIES_1])
    for bar, v in zip(bars, values2):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, v * 1.02, f"{v:,.0f}",
            ha="center", color=style.INK, fontsize=10, fontweight="bold",
        )
    ax2.set_xticks(
        range(2), ["Assumed\na = 0.85", f"Measured\na = {results['accuracy']:.2f}"]
    )
    ax2.set_ylim(0, max(values2) * 1.18)
    ax2.set_ylabel("Expected requests to promote")
    style.titles(
        ax2,
        "What the real channel costs",
        f"Promote 0.50 → 0.55 · {results['config']['sampling_rate']:.0%} sampling",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.14, wspace=0.32)
    fig.savefig(out_dir / "reward_judge_study.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    results = asyncio.run(run(args.seed, args.max_pairs))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reward_judge_study.json").write_text(json.dumps(results, indent=2))
    chart(results, args.out)
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()
