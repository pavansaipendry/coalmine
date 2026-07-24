"""Local reward-model judge: the zero-cost, no-API production sensor.

A sequence-classification reward model scores (question, answer) pairs; the
pairwise verdict is whichever response scores higher (tie inside an optional
margin). Two properties make this an attractive judge besides costing
nothing: scores are computed per response *independently*, so the judge has
no position bias by construction, and inference is deterministic — the same
pair always gets the same verdict. It runs on CPU (per the no-local-GPU
rule); its accuracy is measured against real human preference votes in the
reward_judge_study experiment, not assumed.

`score_fn` is injectable for tests; the default lazy-loads an OpenAssistant
DeBERTa reward model via transformers on first use.
"""

import asyncio
from typing import Callable

import numpy as np

from coalmine.judging.judges import FIRST, SECOND, TIE

DEFAULT_MODELS = (
    "OpenAssistant/reward-model-deberta-v3-base",
    "OpenAssistant/reward-model-deberta-v3-large-v2",
)


def load_reward_scorer(model_name: str | None = None) -> tuple[Callable, str]:
    """Build a score_fn(question, answer) -> float from a HF reward model.
    Falls back through DEFAULT_MODELS if the first is unavailable."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    candidates = [model_name] if model_name else list(DEFAULT_MODELS)
    last_error: Exception | None = None
    for name in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name)
            model.eval()

            def score(question: str, answer: str) -> float:
                inputs = tokenizer(
                    question, answer, return_tensors="pt", truncation=True, max_length=512
                )
                with torch.no_grad():
                    return float(model(**inputs).logits[0].squeeze())

            return score, name
        except Exception as e:  # model unavailable / download failure
            last_error = e
    raise RuntimeError(f"no reward model loadable from {candidates}") from last_error


class RewardModelJudge:
    def __init__(
        self,
        score_fn: Callable[[str, str], float] | None = None,
        judge_id: str | None = None,
        tie_margin: float = 0.0,
    ):
        if score_fn is None:
            score_fn, model_name = load_reward_scorer()
            judge_id = judge_id or f"rm:{model_name.split('/')[-1]}"
        self.score_fn = score_fn
        self.judge_id = judge_id or "rm:injected"
        self.tie_margin = tie_margin

    async def compare(
        self,
        query: str,
        first: dict,
        second: dict,
        rng: np.random.Generator | None = None,
    ) -> str:
        del rng  # deterministic judge; no simulation randomness
        score_first = await asyncio.to_thread(self.score_fn, query, first["text"])
        score_second = await asyncio.to_thread(self.score_fn, query, second["text"])
        if abs(score_first - score_second) <= self.tie_margin:
            return TIE
        return FIRST if score_first > score_second else SECOND
