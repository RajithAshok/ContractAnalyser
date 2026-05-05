"""Tests for Pydantic models and risk scoring logic."""
import pytest
from src.models import Clause, ExtractionResult, RiskLevel, ClauseDiff, ComparisonResult


def make_clause(clause_type="Liability", risk_score=8, risk_level=RiskLevel.HIGH):
    return Clause(
        clause_type=clause_type,
        text="Provider's total liability shall not exceed $100.",
        section_ref="§3",
        risk_score=risk_score,
        risk_level=risk_level,
        risk_reason="Extremely low liability cap unrelated to fees paid.",
    )


def make_result(doc_name="test_saas", n_high=2, n_medium=1, n_low=2):
    clauses = (
        [make_clause("Liability", 9, RiskLevel.HIGH) for _ in range(n_high)]
        + [make_clause("Payment", 5, RiskLevel.MEDIUM) for _ in range(n_medium)]
        + [make_clause("Governing Law", 2, RiskLevel.LOW) for _ in range(n_low)]
    )
    return ExtractionResult(
        document_name=doc_name,
        clauses=clauses,
        summary="A SaaS agreement with several one-sided clauses.",
        overall_risk=RiskLevel.HIGH,
        overall_risk_score=7.5,
    )


class TestClauseModel:
    def test_valid_clause(self):
        c = make_clause()
        assert c.clause_type == "Liability"
        assert 0 <= c.risk_score <= 10

    def test_risk_score_bounds(self):
        with pytest.raises(Exception):
            Clause(clause_type="X", text="y", risk_score=11, risk_level=RiskLevel.LOW)
        with pytest.raises(Exception):
            Clause(clause_type="X", text="y", risk_score=-1, risk_level=RiskLevel.LOW)

    def test_optional_fields(self):
        c = Clause(clause_type="Other", text="some text", risk_score=0, risk_level=RiskLevel.LOW)
        assert c.section_ref is None
        assert c.risk_reason is None


class TestExtractionResult:
    def test_result_creation(self):
        result = make_result()
        assert result.document_name == "test_saas"
        assert len(result.clauses) == 5

    def test_overall_risk_score(self):
        result = make_result()
        assert 0 <= result.overall_risk_score <= 10


class TestRiskScorer:
    def test_build_risk_report_structure(self):
        from src.agents.risk_scorer import build_risk_report
        result = make_result(n_high=2, n_medium=1, n_low=2)
        report = build_risk_report(result)

        assert "counts" in report
        assert report["counts"]["high"] == 2
        assert report["counts"]["medium"] == 1
        assert report["counts"]["low"] == 2
        assert report["counts"]["total"] == 5

    def test_recommendations_generated(self):
        from src.agents.risk_scorer import build_risk_report
        result = make_result(n_high=1)
        report = build_risk_report(result)
        assert isinstance(report["recommendations"], list)

    def test_top_risks_ordered(self):
        from src.agents.risk_scorer import build_risk_report
        result = make_result(n_high=2, n_medium=2, n_low=1)
        report = build_risk_report(result)
        top = report["top_risks"]
        scores = [c.risk_score for c in top]
        assert scores == sorted(scores, reverse=True)

    def test_no_clauses(self):
        from src.agents.risk_scorer import build_risk_report
        result = ExtractionResult(
            document_name="empty",
            clauses=[],
            summary="Empty.",
            overall_risk=RiskLevel.LOW,
            overall_risk_score=0,
        )
        report = build_risk_report(result)
        assert report["counts"]["total"] == 0
        assert report["recommendations"] == []
