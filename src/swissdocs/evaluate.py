"""Retrieval metrics and a lightweight extractive faithfulness check.

Generation-quality metrics that need a live LLM are computed only when a
backend other than ``extractive`` is requested. This module never invents a
score: every number is computed from the gold file and the retriever output.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from swissdocs.models import Answer, Chunk


@dataclass(frozen=True)
class GoldItem:
    id: str
    language: str
    question: str
    gold_articles: tuple[str, ...]
    expect_insufficient: bool = False
    probe_type: str = ""

    @classmethod
    def from_obj(cls, obj: dict[str, object]) -> "GoldItem":
        articles = obj.get("gold_articles", [])
        if not isinstance(articles, list):
            raise ValueError(f"gold_articles must be a list in {obj}")
        return cls(
            id=str(obj["id"]),
            language=str(obj["language"]),
            question=str(obj["question"]),
            gold_articles=tuple(str(item) for item in articles),
            expect_insufficient=bool(obj.get("expect_insufficient", False)),
            probe_type=str(obj.get("probe_type", "")),
        )


def load_gold(path: Path) -> list[GoldItem]:
    items: list[GoldItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(GoldItem.from_obj(json.loads(line)))
    return items


def first_relevant_rank(retrieved_keys: list[str], gold: set[str]) -> int | None:
    for rank, key in enumerate(retrieved_keys, start=1):
        if key in gold:
            return rank
    return None


def hit_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    total = 0.0
    for rank in ranks:
        if rank is not None:
            total += 1.0 / rank
    return total / len(ranks)


def summarise(ranks: list[int | None]) -> dict[str, float | int]:
    return {
        "n": len(ranks),
        "hit_rate_at_1": round(hit_at_k(ranks, 1), 4),
        "hit_rate_at_5": round(hit_at_k(ranks, 5), 4),
        "mrr": round(mean_reciprocal_rank(ranks), 4),
    }


def article_keys(chunks: list[Chunk]) -> set[str]:
    return {chunk.article_key() for chunk in chunks}


def citation_hit(answer: Answer, gold: set[str]) -> bool:
    cited = {f"{item.law_sr}:{item.language}:{item.article}" for item in answer.citations}
    return bool(cited & gold)


def group_metrics(
    rows: list[dict[str, object]],
    rank_field: str,
) -> dict[str, dict[str, float | int]]:
    by_lang: dict[str, list[int | None]] = defaultdict(list)
    all_ranks: list[int | None] = []
    for row in rows:
        rank = row[rank_field]
        value = int(rank) if isinstance(rank, int) else None
        by_lang[str(row["language"])].append(value)
        all_ranks.append(value)
    report = {language: summarise(ranks) for language, ranks in sorted(by_lang.items())}
    report["all"] = summarise(all_ranks)
    return report
