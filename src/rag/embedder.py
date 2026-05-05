"""
Local embedding using sentence-transformers.
No API key, no internet required after first model download (~90MB).
"""
from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"  # 90MB, fast, good quality for retrieval


@lru_cache(maxsize=1)
def _get_model() -> "SentenceTransformer":
    """Load the embedding model once and cache it."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors for the given texts."""
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
