from __future__ import annotations

from typing import List
import numpy as np


def select_by_margin(scores: np.ndarray, delta: float) -> List[int]:
    """
    Return indices whose score is within delta of the max score.
    """
    max_score = float(scores.max())
    order = np.argsort(-scores)  # descending
    return [int(i) for i in order if (max_score - float(scores[i])) <= delta]