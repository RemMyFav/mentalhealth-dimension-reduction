"""Margin-based selection of dimensions for question mapping."""
from __future__ import annotations

from typing import List
import numpy as np


# -------------------------------------------------
# Margin-based Selection
# -------------------------------------------------

def select_by_margin(scores: np.ndarray, delta: float) -> List[int]:
    """Select indices where score is within delta of the maximum.

    Selects all dimensions whose score is within the margin delta of the
    highest-scoring dimension. This enables multi-label assignment where
    a question can belong to multiple related dimensions.

    Args:
        scores: Array of similarity scores for each dimension.
        delta: Margin threshold. Scores within delta of max are selected.

    Returns:
        List of indices corresponding to selected dimensions.
    """
    max_score = float(scores.max())
    order = np.argsort(-scores)  # descending
    return [int(i) for i in order if (max_score - float(scores[i])) <= delta]