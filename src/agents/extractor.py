"""
Clause extraction agent.

Reads the full contract text and returns structured Clause objects.
Uses JSON-mode prompting to enforce schema compliance.
"""
from __future__ import annotations
import json
from src.models import Clause, ExtractionResult, RiskLevel
from src import ollama_client


SYSTEM_PROMPT = """You are a senior contracts attorney. Your job is to analyze legal contracts
and extract all significant clauses in structured JSON format.

For each clause you find, produce:
- clause_type: one of [Confidentiality, Termination, Liability, IP Assignment, Payment,
  Non-Compete, Non-Solicitation, Governing Law, Arbitration, Indemnification, SLA,
  Data Privacy, Warranty, Force Majeure, Amendment, Other]
- text: the key clause text (verbatim or close paraphrase, max 300 chars)
- section_ref: section number/heading if visible
- risk_score: integer 0-10 (0=no risk, 10=extremely unfavorable for the reviewing party)
- risk_level: "low" (0-3), "medium" (4-6), or "high" (7-10)
- risk_reason: one sentence explaining risk if score >= 4, else null

Be thorough. Flag aggressive clauses like: uncapped liability, perpetual IP assignment,
broad non-competes, unilateral amendment rights, mandatory arbitration with class-action waiver.
"""


def extract_clauses(contract_text: str, doc_name: str) -> ExtractionResult:
    """
    Run the extraction agent on the given contract text.
    Returns an ExtractionResult with all clauses and an overall risk assessment.
    """
    # Truncate very long contracts to fit context window
    truncated = contract_text[:12000] if len(contract_text) > 12000 else contract_text

    prompt = f"""Analyze this contract and extract all significant clauses.

CONTRACT TEXT:
{truncated}

Return a JSON object with this exact schema:
{{
  "clauses": [
    {{
      "clause_type": "string",
      "text": "string (max 300 chars)",
      "section_ref": "string or null",
      "risk_score": integer 0-10,
      "risk_level": "low" | "medium" | "high",
      "risk_reason": "string or null"
    }}
  ],
  "summary": "2-3 sentence plain-English summary of what this contract is and its key terms",
  "overall_risk": "low" | "medium" | "high",
  "overall_risk_score": float 0-10
}}
"""

    raw = ollama_client.chat(prompt, system=SYSTEM_PROMPT, expect_json=True, temperature=0.05)
    data = ollama_client.parse_json_response(raw)

    # Handle model returning a bare list instead of the expected object
    if isinstance(data, list):
        data = {"clauses": data, "summary": "", "overall_risk": "low", "overall_risk_score": 0}

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        raw_clauses = []

    clauses = []
    for c in raw_clauses:
        # Skip if the model returned a string instead of a dict
        if not isinstance(c, dict):
            continue
        # Clamp risk_score to valid range
        c["risk_score"] = max(0, min(10, int(c.get("risk_score", 0))))
        try:
            clauses.append(Clause(**c))
        except Exception:
            pass  # Skip malformed clause rather than crashing

    overall_score = float(data.get("overall_risk_score", 0))
    overall_score = max(0.0, min(10.0, overall_score))

    try:
        overall_risk = RiskLevel(data.get("overall_risk", "low"))
    except ValueError:
        overall_risk = RiskLevel.LOW

    return ExtractionResult(
        document_name=doc_name,
        clauses=clauses,
        summary=data.get("summary", ""),
        overall_risk=overall_risk,
        overall_risk_score=round(overall_score, 1),
    )