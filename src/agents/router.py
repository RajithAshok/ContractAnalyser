"""
Router agent — the orchestrator.

Reads the user's query, decides which tool to invoke, calls it,
and returns a structured result. This is the main entry point
for the multi-agent system.
"""
from __future__ import annotations
from enum import Enum
from src import ollama_client


class Intent(str, Enum):
    EXTRACT = "extract"
    RISK = "risk"
    COMPARE = "compare"
    QA = "qa"
    SUMMARIZE = "summarize"
    UNKNOWN = "unknown"


INTENT_SYSTEM = """You classify user queries about legal contracts into one of these intents:
- extract: user wants to see/list the clauses in a contract
- risk: user wants a risk assessment, risk score, or to know what's dangerous
- compare: user wants to compare two contracts against each other
- qa: user wants to ask a factual question about the contract content
- summarize: user wants a plain-English summary or overview
- unknown: none of the above

Respond with ONLY the intent word, nothing else."""


def classify_intent(query: str) -> Intent:
    """Use the LLM to classify the user's intent."""
    raw = ollama_client.chat(
        query,
        system=INTENT_SYSTEM,
        temperature=0.0,
    ).strip().lower()

    for intent in Intent:
        if intent.value in raw:
            return intent
    return Intent.UNKNOWN


def route(
    query: str,
    doc_name: str | None = None,
    doc_name_b: str | None = None,
    loaded_docs: dict | None = None,
) -> dict:
    """
    Main routing function. Classifies the query and dispatches to the right agent.

    Parameters
    ----------
    query       : user's natural language request
    doc_name    : primary document name (if loaded)
    doc_name_b  : second document name (for comparison)
    loaded_docs : dict mapping doc_name -> ExtractionResult (cache)

    Returns a dict with keys: intent, result, message
    """
    # Lazy imports to avoid circular deps
    from src.agents.extractor import extract_clauses
    from src.agents.risk_scorer import build_risk_report
    from src.agents.comparator import compare_documents
    from src.agents.qa import answer_question
    from src.rag.retriever import get_full_document_text

    intent = classify_intent(query)
    loaded_docs = loaded_docs or {}

    if intent == Intent.UNKNOWN:
        # Default to Q&A if we have a document loaded
        intent = Intent.QA if doc_name else Intent.UNKNOWN

    # ── Extract / Risk / Summarize ─────────────────────────────────────
    if intent in (Intent.EXTRACT, Intent.RISK, Intent.SUMMARIZE):
        if not doc_name:
            return {"intent": intent, "result": None, "message": "No document loaded. Load a contract first."}

        if doc_name not in loaded_docs:
            text = get_full_document_text(doc_name)
            if not text:
                return {"intent": intent, "result": None, "message": f"Document '{doc_name}' not found in index."}
            result = extract_clauses(text, doc_name)
            loaded_docs[doc_name] = result
        else:
            result = loaded_docs[doc_name]

        if intent == Intent.RISK:
            risk_report = build_risk_report(result)
            return {"intent": intent, "result": risk_report, "message": "Risk analysis complete."}
        else:
            return {"intent": intent, "result": result, "message": "Extraction complete."}

    # ── Compare ────────────────────────────────────────────────────────
    if intent == Intent.COMPARE:
        if not doc_name or not doc_name_b:
            return {"intent": intent, "result": None, "message": "Two documents required for comparison."}

        for name in (doc_name, doc_name_b):
            if name not in loaded_docs:
                text = get_full_document_text(name)
                if not text:
                    return {"intent": intent, "result": None, "message": f"Document '{name}' not found."}
                loaded_docs[name] = extract_clauses(text, name)

        comparison = compare_documents(loaded_docs[doc_name], loaded_docs[doc_name_b])
        return {"intent": intent, "result": comparison, "message": "Comparison complete."}

    # ── Q&A ────────────────────────────────────────────────────────────
    if intent == Intent.QA:
        if not doc_name:
            return {"intent": intent, "result": None, "message": "No document loaded. Load a contract first."}
        qa_result = answer_question(query, doc_name)
        return {"intent": intent, "result": qa_result, "message": "Answer retrieved."}

    return {"intent": Intent.UNKNOWN, "result": None, "message": "I didn't understand that. Try: extract, risk, compare, or ask a question."}
