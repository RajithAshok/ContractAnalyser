"""
ChromaDB-backed vector store for contract chunks.

Uses a persistent on-disk collection so you don't re-embed on every run.
Supports filtering by document name for per-contract retrieval.
"""
from __future__ import annotations
import chromadb
from chromadb.config import Settings
from pathlib import Path
from src.models import RAGChunk
from src.rag.embedder import embed_texts, embed_query


DB_PATH = Path(".chroma_db")
COLLECTION_NAME = "contracts"


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[RAGChunk]) -> None:
    """Embed and store chunks in ChromaDB."""
    if not chunks:
        return

    collection = _get_collection()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "doc_name": c.doc_name,
                "section_ref": c.section_ref or "",
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ],
    )


def retrieve(
    query: str,
    n_results: int = 5,
    doc_name: str | None = None,
) -> list[dict]:
    """
    Semantic search over indexed chunks.

    Returns a list of dicts with keys: text, doc_name, section_ref, score.
    Optionally filter to a specific document.
    """
    collection = _get_collection()

    # Check if collection has any documents
    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)
    where = {"doc_name": doc_name} if doc_name else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": text,
                "doc_name": meta["doc_name"],
                "section_ref": meta.get("section_ref") or None,
                "score": round(1 - dist, 4),  # cosine distance → similarity
            }
        )

    return chunks


def delete_document(doc_name: str) -> int:
    """Remove all chunks belonging to a document. Returns count deleted."""
    collection = _get_collection()
    existing = collection.get(where={"doc_name": doc_name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        return len(existing["ids"])
    return 0


def list_documents() -> list[str]:
    """Return unique document names currently indexed."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    return list({m["doc_name"] for m in results["metadatas"]})


def get_full_document_text(doc_name: str) -> str:
    """Reconstruct document text from stored chunks, ordered by chunk_index."""
    collection = _get_collection()
    results = collection.get(
        where={"doc_name": doc_name},
        include=["documents", "metadatas"],
    )
    if not results["ids"]:
        return ""
    paired = sorted(
        zip(results["metadatas"], results["documents"]),
        key=lambda x: x[0]["chunk_index"],
    )
    return "\n\n".join(doc for _, doc in paired)
