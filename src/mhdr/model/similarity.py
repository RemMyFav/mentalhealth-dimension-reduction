from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between rows of A (N,D) and rows of B (M,D) -> (N,M)
    """
    eps = 1e-12
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return A_norm @ B_norm.T

