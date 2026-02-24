"""Process human survey data from wide format to tidy format with qid mapping."""
from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_META_COLS = ["ID", "Start time", "Completion time", "Email"]


# -------------------------------------------------
# Text Normalization
# -------------------------------------------------

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


# -------------------------------------------------
# Column Operations
# -------------------------------------------------

def drop_unneeded_columns(df: pd.DataFrame, cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Drop metadata and unneeded columns from raw survey DataFrame.

    Args:
        df: Input DataFrame.
        cols: List of column names to drop. Defaults to DEFAULT_META_COLS.

    Returns:
        DataFrame with specified columns removed.
    """
    cols = DEFAULT_META_COLS if cols is None else cols
    return df.drop(columns=cols, errors="ignore")


# -------------------------------------------------
# QID Mapping
# -------------------------------------------------

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


def find_duplicated_questions(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
) -> pd.DataFrame:
    """Identify duplicated questions based on normalized text.

    Args:
        df: Input DataFrame.
        text_col: Column containing question text.

    Returns:
        DataFrame with columns:
            - text_norm: Normalized text
            - count: Number of occurrences (>=2 indicates duplication)

    Raises:
        ValueError: If text_col is not in the DataFrame.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found.")

    tmp = df.copy()
    tmp["text_norm"] = tmp[text_col].astype(str).map(normalize_text)

    counts = (
        tmp["text_norm"]
        .value_counts()
        .rename_axis("text_norm")
        .reset_index(name="count")
    )
    return counts[counts["count"] >= 2].reset_index(drop=True)


def build_source_ids(
    n: int,
    *,
    prefix: str = "Human",
    start: int = 1,
    zero_pad: int = 2,
) -> list[str]:
    """Generate standardized source identifiers.

    Args:
        n: Number of source IDs to generate.
        prefix: Prefix for each ID (e.g., "Human").
        start: Starting number.
        zero_pad: Number of digits to zero-pad.

    Returns:
        List of source ID strings (e.g., ["Human01", "Human02", ...]).
    """
    return [f"{prefix}{i:0{zero_pad}d}" for i in range(start, start + n)]


# -----------------------------
# Extract All Rows with QID
# -----------------------------

def extract_human_all_with_qid(
    df_raw_wide: pd.DataFrame,
    qid_text_dfs: Sequence[pd.DataFrame],
    *,
    meta_cols_to_drop: Optional[list[str]] = None,
    text_col_in_map: str = "text",
    qid_col_in_map: str = "qid",
    source_prefix: str = "Human",
    source_zero_pad: int = 2,
) -> pd.DataFrame:
    """Convert wide-format human survey to tidy format with qid assignment.

    Args:
        df_raw_wide: Wide-format survey data (rows=respondents, cols=questions).
        qid_text_dfs: DataFrames mapping question text to qid.
        meta_cols_to_drop: Metadata columns to exclude.
        text_col_in_map: Text column name in mapping DataFrames.
        qid_col_in_map: QID column name in mapping DataFrames.
        source_prefix: Prefix for source identifiers.
        source_zero_pad: Zero-padding width for source IDs.

    Returns:
        pd.DataFrame with columns:
            - qid: Question identifier (or None if not mapped)
            - text: Question text
            - answer: Respondent's answer
            - source: Source identifier (e.g., Human01)
            - text_norm: Normalized question text
    """
    wide = drop_unneeded_columns(df_raw_wide, cols=meta_cols_to_drop)

    text2qid_map, _conflict_df = build_text2qid_map_from_dfs(
        qid_text_dfs,
        text_col=text_col_in_map,
        qid_col=qid_col_in_map,
    )

    n_resp = wide.shape[0]
    sources = build_source_ids(n_resp, prefix=source_prefix, zero_pad=source_zero_pad)

    wide2 = wide.copy()
    wide2.index = sources

    melted = wide2.reset_index(names="source").melt(
        id_vars="source", var_name="text", value_name="answer"
    )

    melted = melted.dropna(subset=["answer"]).copy()
    melted["answer"] = melted["answer"].astype(str).str.strip()
    melted = melted[melted["answer"].ne("")].reset_index(drop=True)

    melted = assign_qid_from_text_map(melted, text2qid_map, text_col="text", new_col="qid")

    out = (
        melted[["qid", "text", "answer", "source", "text_norm"]]
        .sort_values(["qid", "text", "source"], na_position="last")
        .reset_index(drop=True)
    )
    return out


# -----------------------------
# DUP Helper: Get Dup Norms from Wide Columns
# -----------------------------

def get_dup_norms_from_wide(
    df_raw_wide: pd.DataFrame,
    *,
    meta_cols_to_drop: Optional[list[str]] = None,
) -> set[str]:
    """Return normalized text values for duplicated questions in wide format.

    Args:
        df_raw_wide: Wide-format survey data.
        meta_cols_to_drop: Metadata columns to exclude.

    Returns:
        Set of normalized text strings that appear multiple times.
    """
    wide = drop_unneeded_columns(df_raw_wide, cols=meta_cols_to_drop)
    q_meta = pd.DataFrame({"text": wide.columns.astype(str)})
    dup_stats = find_duplicated_questions(q_meta, text_col="text")
    return set(dup_stats["text_norm"].tolist())


# -----------------------------
# DUP Rows (Kept for Compatibility)
# -----------------------------

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
    """Extract only rows with duplicated questions.

    Args:
        df_raw_wide: Wide-format survey data.
        qid_text_dfs: DataFrames mapping question text to qid.
        meta_cols_to_drop: Metadata columns to exclude.
        text_col_in_map: Text column name in mapping DataFrames.
        qid_col_in_map: QID column name in mapping DataFrames.
        source_prefix: Prefix for source identifiers.
        source_zero_pad: Zero-padding width for source IDs.

    Returns:
        DataFrame with rows for duplicated questions only.
    """
    all_rows = extract_human_all_with_qid(
        df_raw_wide,
        qid_text_dfs,
        meta_cols_to_drop=meta_cols_to_drop,
        text_col_in_map=text_col_in_map,
        qid_col_in_map=qid_col_in_map,
        source_prefix=source_prefix,
        source_zero_pad=source_zero_pad,
    )
    dup_norms = get_dup_norms_from_wide(df_raw_wide, meta_cols_to_drop=meta_cols_to_drop)
    dup_rows = all_rows[all_rows["text_norm"].isin(dup_norms)].copy()
    return dup_rows.reset_index(drop=True)


def extract_dup_questions(dup_rows: pd.DataFrame) -> pd.DataFrame:
    """Extract unique (qid, text) pairs for duplicated questions.

    Args:
        dup_rows: DataFrame with duplicated question rows.

    Returns:
        DataFrame with columns:
            - qid: Question identifier
            - text: Question text
    """
    return (
        dup_rows[["qid", "text"]]
        .drop_duplicates()
        .dropna(subset=["qid"])
        .sort_values("qid")
        .reset_index(drop=True)
    )


# -----------------------------
# Remove Duplicated Questions from All Rows (by QID Only)
# -----------------------------

def remove_dup_questions(
    all_rows: pd.DataFrame,
    dup_rows: pd.DataFrame,
    *,
    qid_col: str = "qid",
) -> pd.DataFrame:
    """Remove duplicated questions from a tidy table by qid.

    Args:
        all_rows: DataFrame containing all rows.
        dup_rows: DataFrame containing duplicated question rows.
        qid_col: Column name for question ID.

    Returns:
        DataFrame with duplicated qids removed.

    Raises:
        ValueError: If qid_col is not found in either DataFrame.
    """

    if qid_col not in all_rows.columns:
        raise ValueError(f"Column '{qid_col}' not found in all_rows.")
    if qid_col not in dup_rows.columns:
        raise ValueError(f"Column '{qid_col}' not found in dup_rows.")

    # collect duplicated qids
    dup_qids = (
        dup_rows[qid_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # remove them from all_rows
    out = all_rows[
        ~all_rows[qid_col].astype(str).isin(dup_qids)
    ].copy()

    return out.reset_index(drop=True)