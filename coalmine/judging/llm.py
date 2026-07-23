"""Anthropic-backed pairwise judge.

The production sensor: Claude Haiku by default (the cost-discipline choice —
the big experiments never call it; it serves the live demo path and the
anchor-set calibration study that estimates the channel parameters).
Structured outputs guarantee a parseable verdict; temperature 0 keeps the
judge as deterministic as the API allows. The judge sees only query and
response texts — never source_pool, config identity, or presentation history.
"""

import json
from typing import Any

import numpy as np

from coalmine.judging.judges import FIRST, SECOND, TIE

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"winner": {"type": "string", "enum": ["1", "2", "tie"]}},
    "required": ["winner"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are an impartial judge comparing two AI assistant responses to the same "
    "question. Pick the response that answers the question more accurately, more "
    "completely, and more directly. Judge only quality — not length, order, or "
    "style. If the responses are equally good, answer tie."
)

_POSITION = {"1": FIRST, "2": SECOND, "tie": TIE}


class JudgeError(RuntimeError):
    pass


class LLMJudge:
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        client: Any = None,
        judge_id: str | None = None,
        max_tokens: int = 64,
    ):
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.judge_id = judge_id or f"llm:{model}"

    async def compare(
        self,
        query: str,
        first: dict,
        second: dict,
        rng: np.random.Generator | None = None,
    ) -> str:
        del rng  # real judges take no simulation randomness
        prompt = (
            f"Question:\n{query}\n\n"
            f"Response 1:\n{first['text']}\n\n"
            f"Response 2:\n{second['text']}\n\n"
            'Which response is better? Answer "1", "2", or "tie".'
        )
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason not in ("end_turn", "stop_sequence"):
            raise JudgeError(f"judge stopped with {response.stop_reason!r}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise JudgeError("judge returned no text block")
        try:
            winner = json.loads(text)["winner"]
            return _POSITION[winner]
        except (json.JSONDecodeError, KeyError) as e:
            raise JudgeError(f"unparseable judge verdict: {text!r}") from e
