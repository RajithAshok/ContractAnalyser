"""
Cross-document comparator agent.

Compares two ExtractionResults and identifies diverging clauses,
which document is more favorable on each dimension, and an overall recommendation.
"""
from __future__ import annotations
from src.models import ExtractionResult, ComparisonResult, ClauseDiff
from src import ollama_client


SYSTEM_PROMPT = """You are a senior contracts attorney comparing two legal documents.
Your job is to identify meaningful differences between clauses in two contracts
and advise which is more favorable for the party reviewing them (typically the recipient).
"""


def compare_documents(
    result_a: ExtractionResult,
    result_b: ExtractionResult,
) -> ComparisonResult:
    """
    Compare two ExtractionResults and return a ComparisonResult.
    Uses the LLM to generate nuanced diff explanations.
    """
    # Build compact clause summaries for the prompt
    def summarize(result: ExtractionResult) -> str:
        lines = []
        for c in result.clauses:
            lines.append(f"[{c.clause_type}] (risk={c.risk_score}/10) {c.text[:200]}")
        return "\n".join(lines)

    prompt = f"""Compare these two contracts clause by clause.

CONTRACT A — {result_a.document_name}:
{summarize(result_a)}

CONTRACT B — {result_b.document_name}:
{summarize(result_b)}

Return a JSON object:
{{
  "diffs": [
    {{
      "clause_type": "string",
      "doc_a_text": "brief description of doc A's position (or null if absent)",
      "doc_b_text": "brief description of doc B's position (or null if absent)",
      "difference": "one sentence explaining how they differ",
      "which_is_better": "doc_a" | "doc_b" | "equal",
      "reason": "one sentence explaining which is more favorable and why"
    }}
  ],
  "recommendation": "2-3 sentence overall recommendation: which contract is more favorable overall and the most important points to negotiate"
}}

Only include clause types where there is a meaningful difference. Skip identical or trivially similar clauses.
"""

    raw = ollama_client.chat(prompt, system=SYSTEM_PROMPT, expect_json=True, temperature=0.1)
    data = ollama_client.parse_json_response(raw)

    diffs = [ClauseDiff(**d) for d in data.get("diffs", [])]

    return ComparisonResult(
        doc_a_name=result_a.document_name,
        doc_b_name=result_b.document_name,
        diffs=diffs,
        recommendation=data.get("recommendation", ""),
    )
