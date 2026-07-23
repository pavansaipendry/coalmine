"""LLMJudge protocol behavior against a fake Anthropic client — verifies the
request shape and verdict parsing without a network call or API key."""

import json
from types import SimpleNamespace

import pytest

from coalmine.judging.llm import JudgeError, LLMJudge


class FakeMessages:
    def __init__(self, reply_json: str, stop_reason: str = "end_turn"):
        self.reply_json = reply_json
        self.stop_reason = stop_reason
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=self.reply_json)],
        )


def _judge(reply: str, stop_reason: str = "end_turn") -> tuple[LLMJudge, FakeMessages]:
    messages = FakeMessages(reply, stop_reason)
    client = SimpleNamespace(messages=messages)
    return LLMJudge(client=client), messages


FIRST_RESP = {"text": "the grounded answer", "source_pool": None}
SECOND_RESP = {"text": "the vague answer", "source_pool": None}


async def test_verdict_mapping():
    for raw, expected in (("1", "first"), ("2", "second"), ("tie", "tie")):
        judge, _ = _judge(json.dumps({"winner": raw}))
        assert await judge.compare("q", FIRST_RESP, SECOND_RESP) == expected


async def test_request_shape():
    judge, messages = _judge(json.dumps({"winner": "1"}))
    await judge.compare("what is x?", FIRST_RESP, SECOND_RESP)
    kwargs = messages.last_kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["temperature"] == 0.0
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    prompt = kwargs["messages"][0]["content"]
    assert "what is x?" in prompt
    assert "the grounded answer" in prompt
    assert "the vague answer" in prompt
    # The judge must never see config identity or ground-truth labels.
    assert "champion" not in prompt and "source_pool" not in prompt


async def test_unparseable_verdict_raises():
    judge, _ = _judge("I think response 1 is better")
    with pytest.raises(JudgeError):
        await judge.compare("q", FIRST_RESP, SECOND_RESP)


async def test_bad_stop_reason_raises():
    judge, _ = _judge(json.dumps({"winner": "1"}), stop_reason="max_tokens")
    with pytest.raises(JudgeError):
        await judge.compare("q", FIRST_RESP, SECOND_RESP)


def test_judge_id_defaults_to_model():
    judge, _ = _judge("{}")
    assert judge.judge_id == "llm:claude-haiku-4-5"
