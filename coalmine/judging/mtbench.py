"""MT-Bench human judgments: real anchors with human ground truth.

LMSYS released ~3.3K expert human pairwise votes over real model responses
(lmsys/mt_bench_human_judgments). Multiple humans vote on many pairs; the
consensus (majority) vote is the ground-truth label, and pairs without a
majority are dropped. This replaces synthetic good/bad pools as the anchor
set: a judge's accuracy measured against these labels is accuracy against
real human preference on real responses — the parameter every simulation in
coalmine takes as input.
"""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class HumanAnchor:
    question: str
    text_a: str
    text_b: str
    label: str  # "a" | "b" — human-consensus winner
    votes_a: int
    votes_b: int
    votes_tie: int


def consensus_anchors(rows: list[dict]) -> list[HumanAnchor]:
    """Aggregate per-vote rows into consensus-labeled pairs.

    Rows may present the same model pair in either (model_a, model_b) order;
    votes are mapped onto a canonical order before counting. Pairs whose
    non-tie votes have no strict majority are dropped.
    """
    grouped: dict[tuple, dict] = defaultdict(
        lambda: {"votes_a": 0, "votes_b": 0, "votes_tie": 0, "texts": None}
    )
    for row in rows:
        models = sorted([row["model_a"], row["model_b"]])
        key = (row["question_id"], models[0], models[1])
        entry = grouped[key]
        flipped = row["model_a"] != models[0]
        text_a, text_b = (
            (row["text_b"], row["text_a"]) if flipped else (row["text_a"], row["text_b"])
        )
        if entry["texts"] is None:
            entry["texts"] = (row["question"], text_a, text_b)
        winner = row["winner"]
        if winner.startswith("tie"):
            entry["votes_tie"] += 1
        elif (winner == "model_a") != flipped:
            entry["votes_a"] += 1
        else:
            entry["votes_b"] += 1

    anchors = []
    for entry in grouped.values():
        question, text_a, text_b = entry["texts"]
        a, b = entry["votes_a"], entry["votes_b"]
        if a == b:
            continue
        anchors.append(
            HumanAnchor(
                question=question,
                text_a=text_a,
                text_b=text_b,
                label="a" if a > b else "b",
                votes_a=a,
                votes_b=b,
                votes_tie=entry["votes_tie"],
            )
        )
    return anchors


def load_mtbench_rows(split: str = "human") -> list[dict]:
    """Download (cached) and flatten first-turn MT-Bench human votes."""
    from datasets import load_dataset

    dataset = load_dataset("lmsys/mt_bench_human_judgments", split=split)
    rows = []
    for row in dataset:
        if row["turn"] != 1:
            continue
        rows.append(
            {
                "question_id": row["question_id"],
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "winner": row["winner"],
                "question": row["conversation_a"][0]["content"],
                "text_a": row["conversation_a"][1]["content"],
                "text_b": row["conversation_b"][1]["content"],
            }
        )
    return rows
