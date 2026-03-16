from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import pandas as pd

from mhdr.dataloader.text2qid import (
    build_text2qid_map_from_dfs,
    assign_qid_from_text_map,
)

DEFAULT_META_COLS = ["ID", "Start time", "Completion time", "Email", "Name"]

DIMENSIONS = [
    "Emotional",
    "Environmental",
    "Financial",
    "Intellectual",
    "Occupational",
    "Physical",
    "Social",
    "Spiritual",
]


def _parse_ranked_labels(answer: str) -> list[str]:
    """Convert 'Social;Emotional;END OF LIST;...' -> ordered label list."""
    if pd.isna(answer):
        return []

    parts = [x.strip() for x in str(answer).split(";")]

    out = []
    for p in parts:
        if p == "END OF LIST":
            break
        if p:
            out.append(p)

    return out


def _rank_to_score_vector(labels: list[str], dimensions: list[str]) -> dict[str, int]:
    """
    Convert ranking list into Borda-style scores:
    rank_1 -> k
    rank_2 -> k-1
    ...
    others -> 0
    """
    dim2idx = {d: i for i, d in enumerate(dimensions)}
    vec = [0] * len(dimensions)

    k = len(labels)

    for i, label in enumerate(labels):
        if label in dim2idx:
            vec[dim2idx[label]] = k - i

    return dict(zip(dimensions, vec))


def load_multilabel_human_data(
    excel_dir: str | Path,
    qid_text_dfs: Sequence[pd.DataFrame],
    *,
    meta_cols_to_drop: Optional[list[str]] = None,
    text_col_in_map: str = "text",
    qid_col_in_map: str = "qid",
) -> pd.DataFrame:
    """
    Load ranked human label data and convert them to Borda-style scores.

    Returns
    -------
    DataFrame columns:
        qid
        text
        Emotional
        Environmental
        Financial
        Intellectual
        Occupational
        Physical
        Social
        Spiritual
    """

    excel_dir = Path(excel_dir)
    meta_cols_to_drop = DEFAULT_META_COLS if meta_cols_to_drop is None else meta_cols_to_drop

    excel_files = sorted(excel_dir.glob("*.xlsx"))
    if not excel_files:
        raise ValueError(f"No Excel files found in {excel_dir}")

    text2qid_map, _ = build_text2qid_map_from_dfs(
        qid_text_dfs,
        text_col=text_col_in_map,
        qid_col=qid_col_in_map,
    )

    rows = []

    for fp in excel_files:

        df = pd.read_excel(fp)

        wide = df.drop(columns=meta_cols_to_drop, errors="ignore")

        melted = wide.melt(
            var_name="text",
            value_name="answer_raw",
        )

        melted = melted.dropna(subset=["answer_raw"]).copy()
        melted["answer_raw"] = melted["answer_raw"].astype(str).str.strip()
        melted = melted[melted["answer_raw"] != ""]

        melted = assign_qid_from_text_map(
            melted,
            text2qid_map,
            text_col="text",
            new_col="qid",
        )

        melted = melted.dropna(subset=["qid"]).copy()

        melted["answer"] = melted["answer_raw"].map(_parse_ranked_labels)

        score_df = pd.DataFrame(
            melted["answer"].apply(
                lambda x: _rank_to_score_vector(x, DIMENSIONS)
            ).tolist()
        )

        out_part = pd.concat(
            [
                melted[["qid", "text"]].reset_index(drop=True),
                score_df.reset_index(drop=True),
            ],
            axis=1,
        )

        rows.append(out_part)

    out = pd.concat(rows, ignore_index=True)

    return out.sort_values(["qid"]).reset_index(drop=True)