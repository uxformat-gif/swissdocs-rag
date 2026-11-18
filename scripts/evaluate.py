#!/usr/bin/env python3
"""Measure retrieval (and, with --generation, answer checks) on the gold set.

Writes data/eval/results.json. Numbers in the README are copied from that file.
"""

import argparse
import json
import logging
from pathlib import Path

from swissdocs.config import get_settings
from swissdocs.evaluate import (
    citation_hit,
    first_relevant_rank,
    group_metrics,
    load_gold,
)
from swissdocs.retrieval.hybrid import reciprocal_rank_fusion
from swissdocs.service import RagService


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/eval/qa.jsonl")
    parser.add_argument("--output", default="data/eval/results.json")
    parser.add_argument(
        "--generation",
        action="store_true",
        help="Also score the configured LLM backend. Without this flag, "
        "answer checks use the extractive baseline only when LLM_BACKEND=extractive.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not args.generation:
        settings.llm_backend = "extractive"
    service = RagService(settings)
    service.prepare()
    assert service.retriever is not None

    items = load_gold(Path(args.gold))
    answerable = [item for item in items if not item.expect_insufficient]
    probes = [item for item in items if item.expect_insufficient]

    indexed_languages = service.retriever.bm25.languages
    rows = []
    for item in answerable:
        language = item.language
        cross = language not in indexed_languages
        dense = service.retriever.dense(
            item.question,
            language=None if cross else language,
            limit=5,
        )
        if cross:
            lexical_rankings = [
                service.retriever.lexical(item.question, language=lang, limit=5)
                for lang in indexed_languages
            ]
            fused_lexical = reciprocal_rank_fusion(lexical_rankings, k=service.settings.rrf_k)
            lexical = sorted(fused_lexical.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
        else:
            lexical = service.retriever.lexical(item.question, language=language, limit=5)
        hybrid = service.retriever.search(
            item.question,
            language=language,
            top_k=5,
            cross_lingual=cross,
        )
        gold = set(item.gold_articles)

        def keys(pairs: list[tuple[str, float]]) -> list[str]:
            found = []
            for doc_id, _score in pairs:
                chunk = service.retriever.by_id[doc_id]
                found.append(chunk.article_key())
            return found

        dense_rank = first_relevant_rank(keys(dense), gold)
        bm25_rank = first_relevant_rank(keys(lexical), gold)
        hybrid_rank = first_relevant_rank([hit.chunk.article_key() for hit in hybrid], gold)
        top = hybrid[0] if hybrid else None
        rows.append(
            {
                "id": item.id,
                "language": language,
                "question": item.question,
                "gold_articles": list(item.gold_articles),
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "hybrid_rank": hybrid_rank,
                "top_dense_score": None if top is None else top.dense_score,
                "top_bm25_score": None if top is None else top.bm25_score,
                "top_article": None if top is None else top.chunk.article_key(),
            }
        )

    answer_rows = []
    for item in items:
        answer = service.ask(item.question, language=item.language)
        cross = item.language not in indexed_languages
        hits = service.retriever.search(
            item.question,
            language=item.language,
            top_k=service.settings.top_k,
            cross_lingual=cross,
        )
        dense_scores = [hit.dense_score or 0.0 for hit in hits]
        bm25_scores = [hit.bm25_score or 0.0 for hit in hits]
        gold = set(item.gold_articles)
        answer_rows.append(
            {
                "id": item.id,
                "language": item.language,
                "expect_insufficient": item.expect_insufficient,
                "probe_type": item.probe_type,
                "insufficient_context": answer.insufficient_context,
                "backend": answer.backend,
                "best_dense_score": round(max(dense_scores), 4) if dense_scores else None,
                "best_bm25_score": round(max(bm25_scores), 4) if bm25_scores else None,
                "citation_hit": citation_hit(answer, gold) if gold else None,
                "citations": [citation.citation for citation in answer.citations],
            }
        )

    in_scope_answers = [row for row in answer_rows if not row["expect_insufficient"]]
    out_scope_answers = [row for row in answer_rows if row["expect_insufficient"]]
    retrieval = {
        "dense": group_metrics(rows, "dense_rank"),
        "bm25": group_metrics(rows, "bm25_rank"),
        "hybrid_rrf": group_metrics(rows, "hybrid_rank"),
    }
    results = {
        "embedding_model": service.embedder.model_name,
        "llm_backend": service.backend_name,
        "corpus_chunks": len(service.chunks),
        "gold_answerable": len(answerable),
        "min_dense_score": settings.min_dense_score,
        "gold_insufficient_probes": len(probes),
        "retrieval": retrieval,
        "extractive_or_generation": {
            "in_scope_citation_hit_rate": _rate(
                [bool(row["citation_hit"]) for row in in_scope_answers]
            ),
            "in_scope_false_abstention_rate": _rate(
                [bool(row["insufficient_context"]) for row in in_scope_answers]
            ),
            "out_of_scope_abstention_rate": _rate(
                [bool(row["insufficient_context"]) for row in out_scope_answers]
            ),
            "off_topic_abstention_rate": _rate(
                [
                    bool(row["insufficient_context"])
                    for row in out_scope_answers
                    if row["probe_type"] == "off_topic"
                ]
            ),
            "near_miss_abstention_rate": _rate(
                [
                    bool(row["insufficient_context"])
                    for row in out_scope_answers
                    if row["probe_type"] == "near_miss"
                ]
            ),
            "note": (
                "Citation hit rate is the share of answerable questions whose "
                "cited articles include a gold article. Abstention uses the "
                "dense+BM25 gate in SwissDocs (see min_dense_score). These are "
                "not RAGAS scores. Run `python scripts/evaluate.py --generation` "
                "with Ollama or an OpenAI-compatible backend for model answers."
            ),
        },
        "per_question": rows,
        "per_answer": answer_rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    destination.write_text(payload, encoding="utf-8")
    summary = {
        "retrieval": retrieval,
        "answers": results["extractive_or_generation"],
    }
    print(json.dumps(summary, indent=2))


def _rate(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return round(sum(1 for flag in flags if flag) / len(flags), 4)


if __name__ == "__main__":
    main()
