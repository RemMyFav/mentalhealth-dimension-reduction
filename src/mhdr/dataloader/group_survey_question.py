"""Load and consolidate survey questions from multiple CSV files."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Union

PathLike = Union[str, Path]


# -------------------------------------------------
# Load Survey Questions
# -------------------------------------------------

def load_questions_from_dir(
    preprocess_dir: PathLike = "./input",
    *,
    text_col: str = "question",
    qid_col: str = "qid",
    source_col: str = "source",
) -> pd.DataFrame:
    """Load all CSVs from a directory into a single question table.

    Args:
        preprocess_dir: Directory containing CSV files.
        text_col: Column name for question text in each CSV.
        qid_col: Column name for question ID in each CSV.
        source_col: Fallback column for qid if qid_col is missing.

    Returns:
        pd.DataFrame with columns:
            - qid: Unique question identifier
            - text: Question text
            - dataset: Source filename (without extension)

    Raises:
        FileNotFoundError: If the input directory does not exist.
        ValueError: If a CSV is missing required columns.
        RuntimeError: If no CSV files are found in the directory.
    """
    preprocess_dir = Path(preprocess_dir)
    if not preprocess_dir.exists():
        raise FileNotFoundError(f"{preprocess_dir} not found")

    rows = []
    for csv_path in sorted(preprocess_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)

        if text_col not in df.columns:
            raise ValueError(f"{csv_path.name} missing column: {text_col}")

        # choose qid: prefer qid_col, otherwise fallback to source_col
        if qid_col in df.columns:
            qid_series = df[qid_col]
        elif source_col in df.columns:
            qid_series = df[source_col]
        else:
            raise ValueError(f"{csv_path.name} must contain '{qid_col}' or '{source_col}'")

        out = pd.DataFrame({
            "qid": qid_series.astype(str).str.strip(),
            "text": df[text_col].fillna("").astype(str).str.strip(),
            "dataset": csv_path.stem,
        })

        out = out[out["text"].str.len() > 0]
        rows.append(out)

    if not rows:
        raise RuntimeError("No CSV files found in preprocess directory")

    merged = (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates(subset=["qid"], keep="first")
        .reset_index(drop=True)
    )
    return merged