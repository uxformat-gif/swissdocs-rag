import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_quotes_the_measured_hybrid_mrr() -> None:
    """The README table has to match data/eval/results.json. No rounded marketing."""
    results = json.loads((ROOT / "data" / "eval" / "results.json").read_text(encoding="utf-8"))
    hybrid = results["retrieval"]["hybrid_rrf"]["all"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert results["embedding_model"] in readme
    assert f"{hybrid['mrr']:.4f}" in readme
    assert f"{hybrid['hit_rate_at_1']:.4f}" in readme
    assert f"{hybrid['hit_rate_at_5']:.4f}" in readme
    citation = results["extractive_or_generation"]["in_scope_citation_hit_rate"]
    assert f"{citation:.4f}" in readme
