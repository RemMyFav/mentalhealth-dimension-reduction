from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Union

PathLike = Union[str, Path]

def load_questions_from_dir(
    preprocess_dir: PathLike = "./input",
    *,
    text_col: str = "question",
    qid_col: str = "qid",
    source_col: str = "source",
) -> pd.DataFrame:
    """
    Load all CSVs from preprocess_dir into a single question table.

    Expected columns per CSV: at least [text_col] and one of [qid_col] or [source_col].
    Output columns: [qid, dataset, text]
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
            "text": df[text_col].map(process_text),
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

def process_text(text: object) -> str:
    """
    Unified text processing pipeline.

    Current behavior:
        - Strip whitespace
        - Ensure the string is wrapped in double quotes (")
        - Escape internal double quotes (CSV-safe)

    Future extensions:
        - normalization
        - lowercasing
        - punctuation cleanup
        - etc.
    """
    s = "" if text is None else str(text)
    s = s.strip()

    # escape internal quotes
    s = s.replace('"', '""')

    # ensure wrapped in quotes
    if not (len(s) >= 2 and s.startswith('"') and s.endswith('"')):
        s = f'"{s}"'

    return s