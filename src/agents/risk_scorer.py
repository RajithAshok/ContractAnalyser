"""
Risk scoring agent.

Takes an ExtractionResult and produces a detailed risk report,
grouping findings by severity and adding actionable recommendations.
"""
from __future__ import annotations
from src.models import ExtractionResult, Clause, RiskLevel


RISK_EMOJIS = {RiskLevel.HIGH: "🔴", RiskLevel.MEDIUM: "🟡", RiskLevel.LOW: "🟢"}


def build_risk_report(result: ExtractionResult) -> dict:
    """
    Organize clauses into a risk report with severity buckets and stats.
    Returns a dict suitable for display in CLI or web UI.
    """
    high = [c for c in result.clauses if c.risk_level == RiskLevel.HIGH]
    medium = [c for c in result.clauses if c.risk_level == RiskLevel.MEDIUM]
    low = [c for c in result.clauses if c.risk_level == RiskLevel.LOW]

    top_risks = sorted(result.clauses, key=lambda c: c.risk_score, reverse=True)[:5]

    return {
        "document_name": result.document_name,
        "overall_risk": result.overall_risk,
        "overall_risk_score": result.overall_risk_score,
        "summary": result.summary,
        "counts": {"high": len(high), "medium": len(medium), "low": len(low), "total": len(result.clauses)},
        "high_risk_clauses": high,
        "medium_risk_clauses": medium,
        "low_risk_clauses": low,
        "top_risks": top_risks,
        "recommendations": _generate_recommendations(high, medium),
    }


def _generate_recommendations(
    high: list[Clause], medium: list[Clause]
) -> list[str]:
    """Generate plain-English negotiation recommendations from risky clauses."""
    recs = []
    seen_types = set()

    for clause in high:
        if clause.clause_type not in seen_types:
            seen_types.add(clause.clause_type)
            rec = _rec_for_clause(clause)
            if rec:
                recs.append(rec)

    for clause in medium:
        if clause.clause_type not in seen_types and len(recs) < 6:
            seen_types.add(clause.clause_type)
            rec = _rec_for_clause(clause)
            if rec:
                recs.append(rec)

    return recs


def _rec_for_clause(clause: Clause) -> str | None:
    templates = {
        "Liability": "Request a mutual liability cap tied to 12 months of fees paid.",
        "Non-Compete": "Narrow the non-compete to specific roles, geography, and duration (6–12 months max).",
        "IP Assignment": "Carve out pre-existing IP and inventions created outside work hours without company resources.",
        "Termination": "Negotiate for-cause termination requirements and a cure period of at least 30 days.",
        "Indemnification": "Request mutual indemnification rather than one-sided obligations.",
        "Arbitration": "Negotiate to preserve the right to seek injunctive relief in court, and remove class-action waiver if present.",
        "Non-Solicitation": "Limit non-solicitation to 12 months post-termination and exclude passive hiring.",
        "Governing Law": "Negotiate for a neutral jurisdiction or your home jurisdiction.",
        "Payment": "Clarify payment terms and dispute resolution process before fees are considered overdue.",
        "Data Privacy": "Ensure breach notification timelines comply with applicable regulations (e.g., GDPR 72hr).",
        "SLA": "Negotiate for meaningful service credits (10–30% of monthly fees) and define downtime clearly.",
        "Confidentiality": "Confirm mutual obligations and clarify permitted disclosures.",
    }
    return templates.get(clause.clause_type) or (
        f"Review {clause.clause_type} clause carefully: {clause.risk_reason}"
        if clause.risk_reason
        else None
    )
