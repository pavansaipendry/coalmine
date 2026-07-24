"""RewardModelJudge protocol behavior with an injected scorer (no model
download), and the MT-Bench consensus aggregation logic (no dataset)."""

import pytest

from coalmine.judging.judges import compare_randomized
from coalmine.judging.mtbench import consensus_anchors
from coalmine.judging.reward import RewardModelJudge

SCORES = {"strong answer": 2.0, "weak answer": -1.0, "middling": 0.5}


def _scorer(question: str, answer: str) -> float:
    return SCORES[answer]


@pytest.fixture
def judge():
    return RewardModelJudge(score_fn=_scorer)


async def test_higher_score_wins_either_position(judge):
    strong = {"text": "strong answer", "source_pool": None}
    weak = {"text": "weak answer", "source_pool": None}
    assert await judge.compare("q", strong, weak) == "first"
    assert await judge.compare("q", weak, strong) == "second"


async def test_tie_margin(judge):
    close = RewardModelJudge(score_fn=_scorer, tie_margin=2.0)
    a = {"text": "strong answer", "source_pool": None}
    b = {"text": "middling", "source_pool": None}
    assert await close.compare("q", a, b) == "tie"
    assert await judge.compare("q", a, b) == "first"


async def test_position_invariant_through_randomizer(judge):
    # Scores are per-response, so the config-space winner must not depend on
    # the presentation order the randomizer picked.
    import numpy as np

    strong = {"text": "strong answer", "source_pool": None}
    weak = {"text": "weak answer", "source_pool": None}
    winners = set()
    for i in range(20):
        result = await compare_randomized(
            judge, "q", strong, weak, np.random.default_rng([1, i])
        )
        winners.add(result.winner)
    assert winners == {"a"}


def _vote(qid, ma, mb, winner, question="q?", ta="ans-a", tb="ans-b"):
    return {
        "question_id": qid,
        "model_a": ma,
        "model_b": mb,
        "winner": winner,
        "question": question,
        "text_a": ta,
        "text_b": tb,
    }


def test_consensus_majority_and_drop():
    rows = [
        _vote(1, "alpha", "beta", "model_a"),
        _vote(1, "alpha", "beta", "model_a"),
        _vote(1, "alpha", "beta", "model_b"),
        _vote(2, "alpha", "beta", "model_a"),
        _vote(2, "alpha", "beta", "model_b"),  # 1-1 -> dropped
        _vote(3, "alpha", "beta", "tie"),
        _vote(3, "alpha", "beta", "model_b"),
    ]
    anchors = consensus_anchors(rows)
    by_votes = {(a.votes_a, a.votes_b): a for a in anchors}
    assert len(anchors) == 2
    assert by_votes[(2, 1)].label == "a"
    assert by_votes[(0, 1)].label == "b"
    assert by_votes[(0, 1)].votes_tie == 1


def test_consensus_normalizes_swapped_presentation():
    # The same pair voted in both (a, b) orders: votes must land on the same
    # canonical side, with texts mapped accordingly.
    rows = [
        _vote(7, "alpha", "beta", "model_a", ta="alpha-text", tb="beta-text"),
        _vote(7, "beta", "alpha", "model_b", ta="beta-text", tb="alpha-text"),
    ]
    anchors = consensus_anchors(rows)
    assert len(anchors) == 1
    assert anchors[0].votes_a == 2 and anchors[0].votes_b == 0
    assert anchors[0].label == "a"
    assert anchors[0].text_a == "alpha-text"
