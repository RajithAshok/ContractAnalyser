from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Clause(BaseModel):
    clause_type: str = Field(description="Category: e.g. Confidentiality, Termination, Liability, IP, Payment, Non-Compete, Governing Law, Arbitration, Indemnification, SLA, Other")
    text: str = Field(description="The verbatim or close-paraphrase clause text")
    section_ref: Optional[str] = Field(default=None, description="Section number or heading, e.g. '§3.2' or 'Section 7'")
    risk_score: int = Field(default=0, ge=0, le=10, description="Risk score 0-10 for the contracting party receiving this contract")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    risk_reason: Optional[str] = Field(default=None, description="One sentence explaining the risk, if any")


class ExtractionResult(BaseModel):
    document_name: str
    clauses: list[Clause]
    summary: str = Field(description="2-3 sentence plain-English executive summary")
    overall_risk: RiskLevel
    overall_risk_score: float


class ClauseDiff(BaseModel):
    clause_type: str
    doc_a_text: Optional[str] = None
    doc_b_text: Optional[str] = None
    difference: str = Field(description="Plain-English explanation of how the two clauses differ")
    which_is_better: str = Field(description="'doc_a', 'doc_b', or 'equal'")
    reason: str


class ComparisonResult(BaseModel):
    doc_a_name: str
    doc_b_name: str
    diffs: list[ClauseDiff]
    recommendation: str = Field(description="Overall recommendation: which contract is more favorable and why")


class RAGChunk(BaseModel):
    chunk_id: str
    doc_name: str
    text: str
    section_ref: Optional[str] = None
    chunk_index: int


class QAResult(BaseModel):
    question: str
    answer: str
    citations: list[str] = Field(description="List of quoted passages from the contract that support the answer")
    confidence: float = Field(ge=0.0, le=1.0)
