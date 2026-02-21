from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_META_COLS = ["ID", "Start time", "Completion time", "Email"]


def normalize_text(text: str) -> str:
    """Normalize question text for matching/duplicate detection."""
    s = str(text).strip()
    s = re.sub(r"(\.\d+)$", "", s)
    s = re.sub(r"(\s+\d+)$", "", s)
    s = re.sub(r"(\s*[\(\[\{]\s*\d+\s*[\)\]\}])$", "", s)
    s = re.sub(r"(\s*[-–—]\s*\d+)$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[ \t]*[.?!]+$", "", s).strip()
    return s


def drop_unneeded_columns(df: pd.DataFrame, cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Drop metadata/unneeded columns from raw survey table."""
    cols = DEFAULT_META_COLS if cols is None else cols
    return df.drop(columns=cols, errors="ignore")


def build_text2qid_map_from_dfs(
    dfs: Sequence[pd.DataFrame],
    *,
    text_col: str = "text",
    qid_col: str = "qid",
) -> Tuple[dict[str, str], pd.DataFrame]:
    """Build a clean merged text_norm -> qid mapping from multiple DataFrames."""
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

    # Align counts safely (avoid relying on .values ordering)
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
    """Assign qid to a DataFrame using a normalized text -> qid mapping."""
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")

    out = df.copy()
    out["text_norm"] = out[text_col].astype(str).map(normalize_text)
    out[new_col] = out["text_norm"].map(text2qid)
    return out


def find_duplicated_questions(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Identify duplicated questions based on normalized text.

    Args:
        df:
            pd.DataFrame with at least:
                - text_col (str)

        text_col:
            Column containing question text.

    Returns:
        pd.DataFrame with columns:
            - text_norm (str)
            - count (int)

        Only includes text_norm values that appear >= 2 times.

    Notes:
        - Duplicate detection is based on normalized text.
        - Does NOT modify original DataFrame.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found.")

    tmp = df.copy()
    tmp["text_norm"] = tmp[text_col].astype(str).map(normalize_text)

    # Stable across pandas versions: guarantees "count" is numeric
    counts = (
        tmp["text_norm"]
        .value_counts()
        .rename_axis("text_norm")
        .reset_index(name="count")
    )

    duplicated = counts[counts["count"] >= 2].reset_index(drop=True)
    return duplicated


def build_source_ids(
    n: int,
    *,
    prefix: str = "Human",
    start: int = 1,
    zero_pad: int = 2,
) -> list[str]:
    """Generate standardized source identifiers (e.g., Human01..Human28)."""
    return [f"{prefix}{i:0{zero_pad}d}" for i in range(start, start + n)]


def extract_human_dup_with_qid(
    df_raw_wide: pd.DataFrame,
    qid_text_dfs: Sequence[pd.DataFrame],
    *,
    meta_cols_to_drop: Optional[list[str]] = None,
    text_col_in_map: str = "text",
    qid_col_in_map: str = "qid",
    source_prefix: str = "Human",
    source_zero_pad: int = 2,
) -> pd.DataFrame:
    """
    Extract duplicated questions from a HUMAN survey (wide table) and assign qid.

    Returns:
        pd.DataFrame with columns:
            - qid (str | None)
            - text (str)
            - answer (str)
            - source (str)
            - text_norm (str)
    """
    # 1) drop meta columns
    wide = drop_unneeded_columns(df_raw_wide, cols=meta_cols_to_drop)

    # 2) build clean text_norm -> qid map
    text2qid_map, _conflict_df = build_text2qid_map_from_dfs(
        qid_text_dfs,
        text_col=text_col_in_map,
        qid_col=qid_col_in_map,
    )

    # 3) find duplicated questions based on the WIDE column names (question texts)
    q_meta = pd.DataFrame({"text": wide.columns.astype(str)})
    dup_stats = find_duplicated_questions(q_meta, text_col="text")
    dup_norms = set(dup_stats["text_norm"].tolist())

    # 4) melt wide -> tidy: (source, text, answer)
    n_resp = wide.shape[0]
    sources = build_source_ids(n_resp, prefix=source_prefix, zero_pad=source_zero_pad)

    wide2 = wide.copy()
    wide2.index = sources

    melted = wide2.reset_index(names="source").melt(
        id_vars="source", var_name="text", value_name="answer"
    )

    # drop empty answers + standardize
    melted = melted.dropna(subset=["answer"]).copy()
    melted["answer"] = melted["answer"].astype(str).str.strip()
    melted = melted[melted["answer"].ne("")].reset_index(drop=True)

    # 5) assign qid (adds text_norm) then filter to duplicated-question rows
    melted = assign_qid_from_text_map(melted, text2qid_map, text_col="text", new_col="qid")
    dup_rows = melted[melted["text_norm"].isin(dup_norms)].copy()

    # final columns / order
    dup_rows = (
        dup_rows[["qid", "text", "answer", "source", "text_norm"]]
        .sort_values(["qid", "text", "source"], na_position="last")
        .reset_index(drop=True)
    )
    return dup_rows