"""
Optional Gradio web interface for the contract analyzer.

Run: python -m src.app
Then open http://localhost:7860
"""
from __future__ import annotations
import gradio as gr
from pathlib import Path

from src.rag.chunker import chunk_contract
from src.rag.retriever import index_chunks, list_documents
from src.agents.extractor import extract_clauses
from src.agents.risk_scorer import build_risk_report
from src.agents.comparator import compare_documents
from src.agents.qa import answer_question
from src.models import RiskLevel

_cache = {}  # doc_name -> ExtractionResult

RISK_COLORS = {RiskLevel.HIGH: "#dc2626", RiskLevel.MEDIUM: "#d97706", RiskLevel.LOW: "#16a34a"}


def _load_text(file_obj) -> tuple[str, str]:
    if file_obj is None:
        return "", "No file uploaded."
    p = Path(file_obj.name)
    if p.suffix.lower() == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(p) as pdf:
                text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            return "", f"PDF error: {e}"
    else:
        text = p.read_text(errors="replace")
    return text, p.stem


def upload_and_index(file_obj):
    text, doc_name = _load_text(file_obj)
    if not text:
        return f"Error: {doc_name}", gr.update(choices=list_documents())
    chunks = chunk_contract(text, doc_name)
    index_chunks(chunks)
    return f"✓ Indexed '{doc_name}' ({len(chunks)} chunks)", gr.update(choices=list_documents())


def run_extract(doc_name):
    if not doc_name:
        return "Select a document first."
    if doc_name not in _cache:
        from src.rag.retriever import get_full_document_text
        text = get_full_document_text(doc_name)
        if not text:
            return f"Document '{doc_name}' not found in index."
        _cache[doc_name] = extract_clauses(text, doc_name)

    result = _cache[doc_name]
    lines = [f"## {result.document_name}\n\n{result.summary}\n"]
    lines.append(f"**Overall risk: {result.overall_risk.value} ({result.overall_risk_score}/10)**\n")
    lines.append("---")
    for c in sorted(result.clauses, key=lambda x: x.risk_score, reverse=True):
        lines.append(f"### {c.clause_type}  ·  {c.risk_score}/10 {c.risk_level.value}")
        if c.section_ref:
            lines.append(f"*{c.section_ref}*")
        lines.append(f"\n{c.text}\n")
        if c.risk_reason:
            lines.append(f"> ⚠ {c.risk_reason}\n")
    return "\n".join(lines)


def run_risk(doc_name):
    if not doc_name:
        return "Select a document first."
    if doc_name not in _cache:
        run_extract(doc_name)

    report = build_risk_report(_cache[doc_name])
    counts = report["counts"]
    lines = [
        f"## Risk Report — {report['document_name']}\n",
        f"{report['summary']}\n",
        f"🔴 High: {counts['high']}   🟡 Medium: {counts['medium']}   🟢 Low: {counts['low']}\n",
        "---\n",
    ]
    if report["recommendations"]:
        lines.append("### Recommendations\n")
        for i, r in enumerate(report["recommendations"], 1):
            lines.append(f"{i}. {r}")
        lines.append("")

    if report["high_risk_clauses"]:
        lines.append("\n### 🔴 High-risk clauses\n")
        for c in report["high_risk_clauses"]:
            lines.append(f"**{c.clause_type}** ({c.section_ref or '—'}): {c.text[:200]}")
            if c.risk_reason:
                lines.append(f"  > {c.risk_reason}\n")
    return "\n".join(lines)


def run_compare(doc_a, doc_b):
    if not doc_a or not doc_b:
        return "Select two documents to compare."
    if doc_a == doc_b:
        return "Select two different documents."
    for name in (doc_a, doc_b):
        if name not in _cache:
            from src.rag.retriever import get_full_document_text
            text = get_full_document_text(name)
            if not text:
                return f"Document '{name}' not found."
            _cache[name] = extract_clauses(text, name)

    result = compare_documents(_cache[doc_a], _cache[doc_b])
    lines = [f"## Comparison: {doc_a} vs {doc_b}\n\n{result.recommendation}\n\n---\n"]
    for d in result.diffs:
        better_label = {"doc_a": f"✓ {doc_a}", "doc_b": f"✓ {doc_b}", "equal": "Equal"}.get(d.which_is_better, "—")
        lines.append(f"### {d.clause_type}  ·  {better_label}")
        lines.append(f"- **{doc_a}:** {d.doc_a_text or 'absent'}")
        lines.append(f"- **{doc_b}:** {d.doc_b_text or 'absent'}")
        lines.append(f"\n_{d.reason}_\n")
    return "\n".join(lines)


def run_qa(question, doc_name):
    if not question.strip():
        return "Enter a question."
    if not doc_name:
        return "Select a document first."
    result = answer_question(question, doc_name)
    lines = [f"**Q:** {result.question}\n\n**A:** {result.answer}\n"]
    if result.citations:
        lines.append("\n**Citations:**")
        for c in result.citations:
            lines.append(f'  > "{c}"')
    lines.append(f"\n*Confidence: {result.confidence:.0%}*")
    return "\n".join(lines)


def build_ui():
    with gr.Blocks(title="Contract Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ⚖ Contract Analyzer\n*Local AI legal document intelligence — powered by Ollama*")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload contract")
                file_input = gr.File(label="Upload .txt or .pdf", file_types=[".txt", ".pdf"])
                upload_btn = gr.Button("Index document", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False, lines=1)
                doc_selector = gr.Dropdown(label="Active document", choices=list_documents(), interactive=True)

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Extract clauses"):
                        extract_btn = gr.Button("Extract all clauses", variant="primary")
                        extract_out = gr.Markdown()

                    with gr.Tab("Risk analysis"):
                        risk_btn = gr.Button("Run risk analysis", variant="primary")
                        risk_out = gr.Markdown()

                    with gr.Tab("Compare docs"):
                        doc_b_selector = gr.Dropdown(label="Compare against", choices=list_documents(), interactive=True)
                        compare_btn = gr.Button("Compare", variant="primary")
                        compare_out = gr.Markdown()

                    with gr.Tab("Ask a question"):
                        question_input = gr.Textbox(label="Your question", placeholder="What is the termination notice period?", lines=2)
                        qa_btn = gr.Button("Ask", variant="primary")
                        qa_out = gr.Markdown()

        def refresh_dropdowns(status, choices):
            return status, gr.update(choices=choices), gr.update(choices=choices)

        upload_btn.click(
            upload_and_index, inputs=[file_input],
            outputs=[upload_status, doc_selector]
        ).then(lambda c: gr.update(choices=c.choices), inputs=[doc_selector], outputs=[doc_b_selector])

        extract_btn.click(run_extract, inputs=[doc_selector], outputs=[extract_out])
        risk_btn.click(run_risk, inputs=[doc_selector], outputs=[risk_out])
        compare_btn.click(run_compare, inputs=[doc_selector, doc_b_selector], outputs=[compare_out])
        qa_btn.click(run_qa, inputs=[question_input, doc_selector], outputs=[qa_out])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
