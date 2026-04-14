from itertools import combinations
from typing import List, Sequence
import pandas as pd
def retrieve_seeds(
    df: pd.DataFrame,
    dimensions: List[str],
    target_dims: Sequence[str],
    *,
    k: int = 5,
) -> pd.DataFrame:
    """
    Strict seed retriever (always clean multi-dimensional).

    Rule:
        min(target_dims) > other_mean

    If no valid seeds exist, returns empty DataFrame.
    """

    target_dims = list(target_dims)
    other_dims = [d for d in dimensions if d not in target_dims]

    out = df.copy()

    # --- scores ---
    out["target_mean"] = out[target_dims].mean(axis=1)

    if other_dims:
        out["other_mean"] = out[other_dims].mean(axis=1)
    else:
        out["other_mean"] = 0.0

    out["target_min"] = out[target_dims].min(axis=1)

    # --- strict filter ---
    mask = out["target_min"] > out["other_mean"]
    out = out[mask].copy()

    # --- early exit ---
    if len(out) == 0:
        return pd.DataFrame(columns=df.columns)

    # --- ranking ---
    out["score"] = out["target_mean"] - out["other_mean"]
    out = out.sort_values("score", ascending=False)

    return out.head(k).reset_index(drop=True)