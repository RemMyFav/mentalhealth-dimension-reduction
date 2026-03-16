from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence, Tuple

import pandas as pd

def normalize_text(text: str) -> str:
    """Normalize question text for matching and duplicate detection.

    Removes trailing numeric suffixes (.1, 2), parentheses, dashes, and
    normalizes whitespace.

    Args:
        text: Original question text.

    Returns:
        Normalized text string.
    """
    s = str(text).strip()
    s = re.sub(r"(\.\d+)$", "", s)
    s = re.sub(r"(\s+\d+)$", "", s)
    s = re.sub(r"(\s*[\(\[\{]\s*\d+\s*[\)\]\}])$", "", s)
    s = re.sub(r"(\s*[-–—]\s*\d+)$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[ \t]*[.?!]+$", "", s).strip()
    return s

def build_text2qid_map_from_dfs(
    dfs: Sequence[pd.DataFrame],
    *,
    text_col: str = "text",
    qid_col: str = "qid",
) -> Tuple[dict[str, str], pd.DataFrame]:
    """Build a normalized text to qid mapping from multiple DataFrames.

    Creates a mapping from normalized question text to qid, identifying
    both clean mappings (one qid per text) and conflicts (multiple qids).

    Args:
        dfs: Sequence of DataFrames with text and qid columns.
        text_col: Column containing question text.
        qid_col: Column containing question IDs.

    Returns:
        Tuple of (clean_map, conflict_df):
            - clean_map: Dict mapping normalized text to qid
            - conflict_df: DataFrame of texts with multiple qids

    Raises:
        ValueError: If required columns are missing.
    """
    all_rows: list[pd.DataFrame] = []

    for df in dfs:
        if text_col not in df.columns or qid_col not in df.columns:
            raise ValueError(f"Missing required columns: {text_col}, {qid_col}")

        tmp = df[[text_col, qid_col]].dropna().copy()
        tmp["text_norm"] = tmp[text_col].astype(str).map(normalize_text)
        tmp[qid_col] = tmp[qid_col].astype(str).str.strip()
        all_rows.append(tmp[["text_norm", qid_col]])

    merged = pd.concat(all_rows, ignore_index=True)

    grouped = (
        merged.groupby("text_norm")[qid_col]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="qids")
    )
    grouped["n_qids"] = grouped["qids"].map(len)

    counts = merged.groupby("text_norm").size().rename("source_count").reset_index()
    grouped = grouped.merge(counts, on="text_norm", how="left")

    clean = grouped[grouped["n_qids"] == 1].copy()
    conflict = grouped[grouped["n_qids"] > 1].copy()

    clean_map = {row["text_norm"]: row["qids"][0] for _, row in clean.iterrows()}
    return clean_map, conflict


def assign_qid_from_text_map(
    df: pd.DataFrame,
    text2qid: Mapping[str, str],
    *,
    text_col: str = "text",
    new_col: str = "qid",
) -> pd.DataFrame:
    """Assign qid to a DataFrame using normalized text to qid mapping.

    Args:
        df: Input DataFrame with text column.
        text2qid: Mapping from normalized text to qid.
        text_col: Column containing question text.
        new_col: Output column name for assigned qid.

    Returns:
        DataFrame with new_col added containing assigned qids.

    Raises:
        ValueError: If text_col is not in the DataFrame.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")

    out = df.copy()
    out["text_norm"] = out[text_col].astype(str).map(normalize_text)
    out[new_col] = out["text_norm"].map(text2qid)
    return out

