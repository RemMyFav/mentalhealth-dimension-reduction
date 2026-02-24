"""Embedding utilities using sentence transformers."""
from __future__ import annotations

from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# Embedding Functions
# -------------------------------------------------

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load a sentence transformer model for generating embeddings.

    Args:
        model_name: Name of the sentence transformer model.
            Defaults to "all-MiniLM-L6-v2".

    Returns:
        Loaded SentenceTransformer model.
    """
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    """Generate embeddings for a sequence of texts.

    Args:
        model: SentenceTransformer model.
        texts: Sequence of text strings to embed.

    Returns:
        numpy.ndarray of shape (n_texts, embedding_dim).
    """
    return model.encode(texts, convert_to_numpy=True)