"""Response pools: pre-generated answers each config serves from.

Pools are built once (synthetically here; from real LLM generations in Phase
3 — same file format, same interface) and replayed deterministically, so
experiments cost nothing per run and every run is reproducible. The "good"
and "bad" pools per query are what the ε-mixture draws from.
"""

import json
from collections import defaultdict
from pathlib import Path

from coalmine.traffic.dataset import Query

QUALITIES = ("good", "bad")


class ResponsePool:
    def __init__(self, data: dict[tuple[str, str], list[str]]):
        # key: (query_id, quality) -> response variants
        self._data = data

    def responses(self, query_id: str, quality: str) -> list[str]:
        try:
            return self._data[(query_id, quality)]
        except KeyError:
            raise KeyError(f"no {quality!r} pool for query {query_id!r}") from None

    @staticmethod
    def build_synthetic(queries: list[Query], variants: int = 4) -> "ResponsePool":
        data: dict[tuple[str, str], list[str]] = {}
        for q in queries:
            data[(q.id, "good")] = [
                f"[{q.topic}] Grounded answer to {q.id}, variant {k}: states the key fact "
                "directly, cites the mechanism, and qualifies its scope."
                for k in range(variants)
            ]
            data[(q.id, "bad")] = [
                f"[{q.topic}] Vague answer to {q.id}, variant {k}: hedges, restates the "
                "question, and omits the key fact entirely."
                for k in range(variants)
            ]
        return ResponsePool(data)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            for (query_id, quality), texts in sorted(self._data.items()):
                f.write(
                    json.dumps({"query_id": query_id, "quality": quality, "responses": texts})
                    + "\n"
                )

    @staticmethod
    def load(path: str | Path) -> "ResponsePool":
        data: dict[tuple[str, str], list[str]] = defaultdict(list)
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    data[(d["query_id"], d["quality"])] = list(d["responses"])
        return ResponsePool(dict(data))
