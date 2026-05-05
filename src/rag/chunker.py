"""
Contract-aware semantic chunker.

Legal contracts have clear structure: numbered sections, headings, and clauses.
This chunker tries to split on section boundaries first, then falls back to
sentence-aware splitting. Each chunk carries metadata about its origin.
"""
from __future__ import annotations
import re
import hashlib
from src.models import RAGChunk


# Patterns that signal the start of a new section
SECTION_PATTERNS = [
    re.compile(r"^\s*(\d+\.)\s+[A-Z]", re.MULTILINE),          # 1. HEADING
    re.compile(r"^\s*([A-Z]{2,}[\s\w]*)\s*$", re.MULTILINE),    # ALL CAPS HEADING
    re.compile(r"^\s*(ARTICLE|SECTION|CLAUSE)\s+\w+", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*([IVX]+\.)\s+[A-Z]", re.MULTILINE),        # Roman numerals
]

MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 100
OVERLAP_CHARS = 150


def chunk_contract(text: str, doc_name: str) -> list[RAGChunk]:
    """
    Split a contract into overlapping chunks with section metadata.
    Returns a list of RAGChunk objects ready for embedding.
    """
    text = _normalize_whitespace(text)
    sections = _split_into_sections(text)

    chunks: list[RAGChunk] = []
    chunk_index = 0

    for section_ref, section_text in sections:
        if len(section_text) <= MAX_CHUNK_CHARS:
            # Section fits in one chunk
            if len(section_text) >= MIN_CHUNK_CHARS:
                chunks.append(_make_chunk(doc_name, section_text, section_ref, chunk_index))
                chunk_index += 1
        else:
            # Section too long — split into overlapping sub-chunks
            sub_chunks = _split_with_overlap(section_text, MAX_CHUNK_CHARS, OVERLAP_CHARS)
            for sub in sub_chunks:
                if len(sub) >= MIN_CHUNK_CHARS:
                    chunks.append(_make_chunk(doc_name, sub, section_ref, chunk_index))
                    chunk_index += 1

    return chunks


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """
    Find section boundaries and split text into (section_ref, section_text) tuples.
    Falls back to treating the whole document as one section.
    """
    # Find all section start positions using all patterns
    boundaries: list[tuple[int, str]] = []

    for pattern in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            ref = m.group(0).strip()[:60]
            boundaries.append((m.start(), ref))

    if not boundaries:
        # No section structure found — treat as one block
        return [(None, text)]

    # Sort by position and deduplicate nearby matches
    boundaries.sort(key=lambda x: x[0])
    deduped: list[tuple[int, str]] = []
    last_pos = -200
    for pos, ref in boundaries:
        if pos - last_pos > 50:
            deduped.append((pos, ref))
            last_pos = pos

    # Build sections from boundaries
    sections = []
    for i, (pos, ref) in enumerate(deduped):
        end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        section_text = text[pos:end].strip()
        if section_text:
            sections.append((ref, section_text))

    # Capture any text before the first section boundary
    if deduped and deduped[0][0] > 50:
        preamble = text[: deduped[0][0]].strip()
        if preamble:
            sections.insert(0, ("Preamble", preamble))

    return sections if sections else [(None, text)]


def _split_with_overlap(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Split a long text into overlapping chunks, trying to break on sentence
    or paragraph boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Try to find a sentence boundary near the end
            for sep in ["\n\n", ".\n", ". ", "\n"]:
                idx = text.rfind(sep, start, end)
                if idx != -1 and idx > start + MIN_CHUNK_CHARS:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start >= len(text) - MIN_CHUNK_CHARS:
            break
    return chunks


def _make_chunk(doc_name: str, text: str, section_ref: str | None, index: int) -> RAGChunk:
    chunk_id = hashlib.md5(f"{doc_name}:{index}:{text[:50]}".encode()).hexdigest()[:12]
    return RAGChunk(
        chunk_id=chunk_id,
        doc_name=doc_name,
        text=text,
        section_ref=section_ref,
        chunk_index=index,
    )


def _normalize_whitespace(text: str) -> str:
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces on lines
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return text.strip()
