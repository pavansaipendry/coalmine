"""Queries: what the traffic simulator replays.

Loads real datasets from JSONL ({"id", "text", "topic"} per line) or builds a
synthetic topic-labeled set for experiments that don't need real text. Topic
labels drive the distribution-shift knob now and the drift monitors later.
"""

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TOPICS = ["science", "history", "coding", "health"]

_TEMPLATES = {
    "science": "Why does {} happen at the physical level?",
    "history": "What were the causes and consequences of event {}?",
    "coding": "How do I fix bug {} in a concurrent system?",
    "health": "What does the evidence say about treatment {}?",
}


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    topic: str


def synthetic_queries(n: int, topics: list[str] | None = None) -> list[Query]:
    """n queries spread evenly across topics, deterministic by construction."""
    topics = topics or DEFAULT_TOPICS
    out = []
    for i in range(n):
        topic = topics[i % len(topics)]
        template = _TEMPLATES.get(topic, "Question about {}?")
        out.append(Query(id=f"q{i:05d}", text=template.format(f"#{i}"), topic=topic))
    return out


def load_jsonl(path: str | Path) -> list[Query]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(Query(id=str(d["id"]), text=d["text"], topic=d.get("topic", "none")))
    return out


def save_jsonl(queries: list[Query], path: str | Path) -> None:
    with open(path, "w") as f:
        for q in queries:
            f.write(json.dumps({"id": q.id, "text": q.text, "topic": q.topic}) + "\n")
