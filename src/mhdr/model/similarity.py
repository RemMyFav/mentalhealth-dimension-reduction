"""Cosine similarity computations for embeddings."""
from __future__ import annotations

import numpy as np


# -------------------------------------------------
# Cosine Similarity
# -------------------------------------------------

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between row vectors of two matrices.

    Args:
        A: First array of shape (N, D).
        B: Second array of shape (M, D).

    Returns:
        Similarity matrix of shape (N, M) where result[i, j] is the
        cosine similarity between A[i] and B[j].
    """
    eps = 1e-12
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return A_norm @ B_norm.T

