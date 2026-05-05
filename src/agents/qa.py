"""
RAG-based Q&A agent.

Retrieves the most relevant contract chunks for a query,
then asks the LLM to answer grounded in that context.
"""
from __future__ import annotations
from src.models import QAResult
from src.rag.retriever import retrieve
from src import ollama_client


SYSTEM_PROMPT = """You are a contracts attorney answering questions about a specific legal document.
Answer ONLY based on the provided contract excerpts. If the answer is not in the excerpts,
say so explicitly. Be precise, cite relevant section numbers when available.
Do not speculate beyond what the contract says."""


def answer_question(question: str, doc_name: str, n_chunks: int = 4) -> QAResult:
    """
    Retrieve relevant chunks from the contract and answer the question.
    Returns a QAResult with the answer and source citations.
    """
    chunks = retrieve(question, n_results=n_chunks, doc_name=doc_name)

    if not chunks:
        return QAResult(
            question=question,
            answer="No relevant content found. Make sure the document is indexed.",
            citations=[],
            confidence=0.0,
        )

    context_blocks = []
    for i, chunk in enumerate(chunks):
        ref = f" ({chunk['section_ref']})" if chunk.get("section_ref") else ""
        context_blocks.append(f"[Excerpt {i+1}{ref}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""Contract excerpts from "{doc_name}":

{context}

---

Question: {question}

Answer the question based strictly on the excerpts above. Then list 1-3 short direct quotes
(under 100 chars each) from the excerpts that support your answer.

Return JSON:
{{
  "answer": "your answer",
  "citations": ["quote 1", "quote 2"],
  "confidence": float 0.0-1.0
}}
"""

    raw = ollama_client.chat(prompt, system=SYSTEM_PROMPT, expect_json=True, temperature=0.1)
    data = ollama_client.parse_json_response(raw)

    return QAResult(
        question=question,
        answer=data.get("answer", ""),
        citations=data.get("citations", []),
        confidence=float(data.get("confidence", 0.5)),
    )
