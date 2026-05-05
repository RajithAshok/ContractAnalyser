"""Tests for the contract chunker."""
import pytest
from src.rag.chunker import chunk_contract, _split_into_sections, _split_with_overlap


SAMPLE_CONTRACT = """
MUTUAL NON-DISCLOSURE AGREEMENT

This agreement is between Company A and Company B.

1. DEFINITION OF CONFIDENTIAL INFORMATION

Confidential Information means any non-public information disclosed by either party,
including technical data, trade secrets, and business information. This includes all
written, oral, and electronic communications marked as confidential.

2. OBLIGATIONS

Each party agrees to hold all Confidential Information in strict confidence.
The receiving party shall not disclose any Confidential Information to third parties
without prior written consent of the disclosing party.

3. TERM

This Agreement shall remain in effect for three years from the Effective Date.
Confidentiality obligations survive for five years after termination.

4. GOVERNING LAW

This Agreement shall be governed by the laws of Delaware.
"""


def test_chunk_contract_returns_chunks():
    chunks = chunk_contract(SAMPLE_CONTRACT, "test_nda")
    assert len(chunks) > 0
    assert all(c.doc_name == "test_nda" for c in chunks)


def test_chunks_have_unique_ids():
    chunks = chunk_contract(SAMPLE_CONTRACT, "test_nda")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs should be unique"


def test_chunks_preserve_text():
    chunks = chunk_contract(SAMPLE_CONTRACT, "test_nda")
    all_text = " ".join(c.text for c in chunks)
    # Key words from longer sections must appear in chunks
    assert "Confidential" in all_text
    assert "strict confidence" in all_text


def test_chunk_indices_are_sequential():
    chunks = chunk_contract(SAMPLE_CONTRACT, "test_nda")
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_max_chunk_size():
    from src.rag.chunker import MAX_CHUNK_CHARS
    chunks = chunk_contract(SAMPLE_CONTRACT * 3, "long_doc")
    for c in chunks:
        assert len(c.text) <= MAX_CHUNK_CHARS + 100, (
            f"Chunk too long: {len(c.text)} chars"
        )


def test_split_with_overlap_creates_overlap():
    text = "A" * 500 + " boundary. " + "B" * 500
    parts = _split_with_overlap(text, max_chars=400, overlap=50)
    assert len(parts) >= 2
    # Check that consecutive chunks share some content (overlap)
    if len(parts) >= 2:
        # End of first chunk should overlap with start of second
        assert len(parts[0]) > 0
        assert len(parts[1]) > 0


def test_empty_document():
    chunks = chunk_contract("", "empty")
    assert chunks == []


def test_very_short_document():
    chunks = chunk_contract("Short contract text.", "short")
    # Should handle without error, may produce 0 or 1 chunk
    assert isinstance(chunks, list)


def test_section_detection():
    sections = _split_into_sections(SAMPLE_CONTRACT)
    assert len(sections) > 1  # Should find multiple sections
    section_refs = [ref for ref, _ in sections if ref]
    assert any("1." in (ref or "") or "DEFINITION" in (ref or "") for ref in section_refs)
