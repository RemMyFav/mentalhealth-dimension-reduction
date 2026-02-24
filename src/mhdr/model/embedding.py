from __future__ import annotations

from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    # sentence-transformers already handles batching internally
    return model.encode(list(texts), convert_to_numpy=True)